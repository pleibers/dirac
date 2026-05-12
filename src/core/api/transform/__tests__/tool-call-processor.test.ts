import "should"
import { processResponsesEvents } from "../../providers/openai-responses-utils"
import { handleResponsesApiStreamResponse } from "../../utils/responses_api_support"
import { getOpenAIToolParams, ToolCallProcessor } from "../tool-call-processor"

function toolCallArguments(chunks: ReturnType<ToolCallProcessor["processToolCallDeltas"]>): string[] {
	return [...chunks].map((chunk) => (chunk.tool_call as any).function.arguments)
}

async function* streamEvents(...events: any[]): AsyncGenerator<any> {
	for (const event of events) {
		yield event
	}
}

async function collectStreamToolCallArguments(stream: AsyncIterable<any>): Promise<string[]> {
	const argumentsDeltas: string[] = []
	for await (const chunk of stream) {
		if (chunk.type === "tool_calls") {
			argumentsDeltas.push(chunk.tool_call.function.arguments)
		}
	}
	return argumentsDeltas
}

describe("ToolCallProcessor", () => {
	it("should preserve strict argument deltas", () => {
		const processor = new ToolCallProcessor()

		const setupChunk = [
			{
				index: 0,
				id: "call_delta",
				function: { name: "read_file" },
			},
		] as any

		;[...processor.processToolCallDeltas(setupChunk)].should.have.length(0)

		const argumentDeltas = ["{", '"path"', ":", '"x"', "}"]
		const emittedArguments = argumentDeltas.flatMap((argumentsDelta) =>
			toolCallArguments(
				processor.processToolCallDeltas([
					{
						index: 0,
						function: { arguments: argumentsDelta },
					},
				] as any),
			),
		)

		emittedArguments.join("").should.equal('{"path":"x"}')
		emittedArguments.should.deepEqual(argumentDeltas)
	})

	it("should emit only the missing suffix for cumulative argument chunks", () => {
		const processor = new ToolCallProcessor()

		;[
			...processor.processToolCallDeltas([
				{
					index: 0,
					id: "call_cumulative",
					function: { name: "read_file" },
				},
			] as any),
		].should.have.length(0)

		const firstResult = toolCallArguments(
			processor.processToolCallDeltas([
				{
					index: 0,
					function: { arguments: '{"path"' },
				},
			] as any),
		)
		const secondResult = toolCallArguments(
			processor.processToolCallDeltas([
				{
					index: 0,
					function: { arguments: '{"path":"x"}' },
				},
			] as any),
		)

		firstResult.should.deepEqual(['{"path"'])
		secondResult.should.deepEqual([':"x"}'])
	})

	it("should ignore repeated complete cumulative argument payloads", () => {
		const processor = new ToolCallProcessor()

		;[
			...processor.processToolCallDeltas([
				{
					index: 0,
					id: "call_repeated",
					function: { name: "read_file" },
				},
			] as any),
		].should.have.length(0)

		toolCallArguments(
			processor.processToolCallDeltas([
				{
					index: 0,
					function: { arguments: '{"path":"x"}' },
				},
			] as any),
		).should.deepEqual(['{"path":"x"}'])
		toolCallArguments(
			processor.processToolCallDeltas([
				{
					index: 0,
					function: { arguments: '{"path":"x"}' },
				},
			] as any),
		).should.deepEqual([])
	})

	it("should preserve tool call id/name for interleaved parallel deltas", () => {
		const processor = new ToolCallProcessor()

		const firstChunk = [
			{
				index: 0,
				id: "call_a",
				function: { name: "read_file" },
			},
			{
				index: 1,
				id: "call_b",
				function: { name: "search_files" },
			},
		] as any

		const secondChunk = [
			{
				index: 1,
				function: { arguments: '{"path":"src"}' },
			},
			{
				index: 0,
				function: { arguments: '{"path":"README.md"}' },
			},
		] as any

		const firstResult = [...processor.processToolCallDeltas(firstChunk)]
		const secondResult = [...processor.processToolCallDeltas(secondChunk)]

		firstResult.should.have.length(0)
		secondResult.should.have.length(2)
		// Intentionally reversed from the setup chunk: output follows incoming
		// argument-delta order, but reconstruction is correct regardless of arrival
		// order because id/name/arguments are matched by tool call index.
		const firstToolCall = secondResult[0]!.tool_call as any
		const secondToolCall = secondResult[1]!.tool_call as any
		firstToolCall.function.id.should.equal("call_b")
		firstToolCall.function.name.should.equal("search_files")
		firstToolCall.function.arguments.should.equal('{"path":"src"}')
		secondToolCall.function.id.should.equal("call_a")
		secondToolCall.function.name.should.equal("read_file")
		secondToolCall.function.arguments.should.equal('{"path":"README.md"}')
	})

	it("should track cumulative argument streams independently by parallel tool call index", () => {
		const processor = new ToolCallProcessor()

		;[
			...processor.processToolCallDeltas([
				{
					index: 0,
					id: "call_a",
					function: { name: "read_file" },
				},
				{
					index: 1,
					id: "call_b",
					function: { name: "search_files" },
				},
			] as any),
		].should.have.length(0)

		const firstResult = [
			...processor.processToolCallDeltas([
				{
					index: 0,
					function: { arguments: '{"path"' },
				},
				{
					index: 1,
					function: { arguments: '{"query"' },
				},
			] as any),
		]
		const secondResult = [
			...processor.processToolCallDeltas([
				{
					index: 1,
					function: { arguments: '{"query":"src"}' },
				},
				{
					index: 0,
					function: { arguments: '{"path":"README.md"}' },
				},
			] as any),
		]

		firstResult.map((chunk) => (chunk.tool_call as any).function.arguments).should.deepEqual(['{"path"', '{"query"'])
		secondResult.map((chunk) => (chunk.tool_call as any).function.arguments).should.deepEqual([':"src"}', ':"README.md"}'])
		;(secondResult[0]!.tool_call as any).function.id.should.equal("call_b")
		;(secondResult[1]!.tool_call as any).function.id.should.equal("call_a")
	})

	it("should clear accumulated state on reset", () => {
		const processor = new ToolCallProcessor()

		const setupChunk = [
			{
				index: 0,
				id: "call_reset",
				function: { name: "read_file" },
			},
		] as any

		const argsChunk = [
			{
				index: 0,
				function: { arguments: '{"path":"after-reset"}' },
			},
		] as any

		;[...processor.processToolCallDeltas(setupChunk)].should.have.length(0)
		processor.reset()
		;[...processor.processToolCallDeltas(argsChunk)].should.have.length(0)

		const newSetupChunk = [
			{
				index: 0,
				id: "call_new",
				function: { name: "write_file" },
			},
		] as any

		const newArgsChunk = [
			{
				index: 0,
				function: { arguments: '{"path":"file.txt"}' },
			},
		] as any

		;[...processor.processToolCallDeltas(newSetupChunk)].should.have.length(0)
		;[...processor.processToolCallDeltas(newArgsChunk)].should.have.length(1)
	})
})

describe("OpenAI Responses tool call argument streaming", () => {
	it("should not duplicate full arguments emitted after cumulative deltas", async () => {
		const argumentsDeltas = await collectStreamToolCallArguments(
			processResponsesEvents(
				streamEvents(
					{
						type: "response.output_item.added",
						item: {
							id: "fc_123",
							type: "function_call",
							call_id: "call_123",
							name: "read_file",
							arguments: "",
						},
					},
					{
						type: "response.function_call_arguments.delta",
						item_id: "fc_123",
						delta: '{"path"',
					},
					{
						type: "response.function_call_arguments.delta",
						item_id: "fc_123",
						delta: '{"path":"x"}',
					},
					{
						type: "response.function_call_arguments.done",
						item_id: "fc_123",
						name: "read_file",
						arguments: '{"path":"x"}',
					},
				),
				{} as any,
			),
		)

		argumentsDeltas.should.deepEqual(['{"path"', ':"x"}'])
	})

	it("should not duplicate full arguments in the legacy Responses handler", async () => {
		const argumentsDeltas = await collectStreamToolCallArguments(
			handleResponsesApiStreamResponse(
				streamEvents(
					{
						type: "response.function_call_arguments.delta",
						item_id: "fc_123",
						delta: '{"path"',
					},
					{
						type: "response.function_call_arguments.delta",
						item_id: "fc_123",
						delta: '{"path":"x"}',
					},
					{
						type: "response.function_call_arguments.done",
						item_id: "fc_123",
						name: "read_file",
						arguments: '{"path":"x"}',
					},
				) as any,
				{} as any,
				async () => 0,
			),
		)

		argumentsDeltas.should.deepEqual(['{"path"', ':"x"}'])
	})
})

describe("getOpenAIToolParams", () => {
	it("should include parallel_tool_calls when enabled", () => {
		const tools = [
			{ type: "function", function: { name: "read_file", description: "", parameters: { type: "object" } } },
		] as any
		const params = getOpenAIToolParams(tools, true) as any

		params.parallel_tool_calls.should.equal(true)
	})

	it("should include parallel_tool_calls=false when disabled by default", () => {
		const tools = [
			{ type: "function", function: { name: "read_file", description: "", parameters: { type: "object" } } },
		] as any
		const params = getOpenAIToolParams(tools, false) as any

		params.parallel_tool_calls.should.equal(false)
	})

	it("should not include parallel_tool_calls when tools are absent", () => {
		const params = getOpenAIToolParams(undefined, false) as any

		params.should.not.have.property("parallel_tool_calls")
	})
})
