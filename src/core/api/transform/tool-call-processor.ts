import type {
	ChatCompletionChunk,
	ChatCompletionToolChoiceOption,
	ChatCompletionTool as OpenAITool,
} from "openai/resources/chat/completions"
import { Logger } from "@/shared/services/Logger"
import type { ApiStreamToolCallsChunk } from "./stream"

interface ToolCallState {
	id: string
	name: string
	arguments: string
}

export function getStreamingArgumentDelta(
	accumulatedArguments: string,
	incomingArguments: unknown,
): { accumulatedArguments: string; argumentDelta: string | undefined } {
	if (incomingArguments === undefined) {
		return { accumulatedArguments, argumentDelta: undefined }
	}

	const incomingArgumentDelta = typeof incomingArguments === "string" ? incomingArguments : JSON.stringify(incomingArguments)

	if (incomingArgumentDelta.startsWith(accumulatedArguments)) {
		const missingSuffix = incomingArgumentDelta.slice(accumulatedArguments.length)
		return {
			accumulatedArguments: incomingArgumentDelta,
			argumentDelta: missingSuffix || undefined,
		}
	}

	return {
		accumulatedArguments: accumulatedArguments + incomingArgumentDelta,
		argumentDelta: incomingArgumentDelta,
	}
}

/**
 * Helper class to process tool call deltas from OpenAI-compatible streaming responses.
 * Handles accumulating tool call ID and name across multiple delta chunks,
 * and yields properly formatted tool call chunks when arguments are received.
 */
export class ToolCallProcessor {
	private toolCallStateByIndex: Map<number, ToolCallState>

	constructor() {
		this.toolCallStateByIndex = new Map()
	}

	/**
	 * Process tool call deltas from a chunk and yield formatted tool call chunks.
	 * @param toolCallDeltas - Array of tool call deltas from the chunk
	 * @yields Formatted tool call chunks ready to be yielded in the API stream
	 */
	*processToolCallDeltas(
		toolCallDeltas: ChatCompletionChunk.Choice.Delta.ToolCall[] | undefined,
	): Generator<ApiStreamToolCallsChunk> {
		if (!toolCallDeltas) {
			return
		}

		for (const [fallbackIndex, toolCallDelta] of toolCallDeltas.entries()) {
			// OpenAI-style streams include an index per tool call. Use iteration order as a fallback.
			const toolCallIndex = toolCallDelta.index ?? fallbackIndex
			const toolCallState = this.getOrCreateToolCallState(toolCallIndex)

			// Accumulate the tool call ID if present
			if (toolCallDelta.id) {
				toolCallState.id = toolCallDelta.id
			}

			// Accumulate web_search type
			if ((toolCallDelta as any).type === "web_search") {
				toolCallState.name = "web_search"
			}

			// Accumulate the function name if present
			if (toolCallDelta.function?.name) {
				Logger.debug(`[ToolCallProcessor] Native Tool Called: ${toolCallDelta.function.name}`)
				toolCallState.name = toolCallDelta.function.name
			}

			// Only yield when we have all required fields: id, name, and arguments
			// Only yield when we have all required fields: id, name, and arguments (or web_search query)
			const normalizedArgumentDelta = this.normalizeArgumentDelta(toolCallState, toolCallDelta.function?.arguments)
			const hasFunctionArgs = normalizedArgumentDelta !== undefined
			const hasWebSearchQuery = (toolCallDelta as any).web_search?.query !== undefined

			if (toolCallState.id && toolCallState.name && (hasFunctionArgs || hasWebSearchQuery)) {
				yield {
					type: "tool_calls",
					tool_call: (toolCallState.name === "web_search"
						? {
								call_id: toolCallState.id,
								type: "web_search",
								web_search: (toolCallDelta as any).web_search || { query: "" },
							}
						: {
								...toolCallDelta,
								function: {
									...toolCallDelta.function,
									id: toolCallState.id,
									name: toolCallState.name,
									arguments: normalizedArgumentDelta,
								},
							}) as any,
				}
			}
		}
	}

	private getOrCreateToolCallState(index: number): ToolCallState {
		const existingState = this.toolCallStateByIndex.get(index)
		if (existingState) {
			return existingState
		}

		const initialState = { id: "", name: "", arguments: "" }
		this.toolCallStateByIndex.set(index, initialState)
		return initialState
	}

	private normalizeArgumentDelta(toolCallState: ToolCallState, incomingArguments: unknown): string | undefined {
		const normalized = getStreamingArgumentDelta(toolCallState.arguments, incomingArguments)
		toolCallState.arguments = normalized.accumulatedArguments
		return normalized.argumentDelta
	}

	/**
	 * Reset the internal state. Call this when starting a new message.
	 */
	reset(): void {
		this.toolCallStateByIndex.clear()
	}

	/**
	 * Get the current accumulated tool call state (useful for debugging).
	 */
	getState(): Record<number, ToolCallState> {
		return Object.fromEntries(this.toolCallStateByIndex.entries())
	}
}

export function getOpenAIToolParams(tools?: OpenAITool[], enableParallelToolCalls = false) {
	if (!tools?.length) {
		return {
			tools: undefined,
		}
	}

	const mappedTools = tools.map((tool) => {
		if (tool.type === "function") {
			return tool
		}
		if ((tool as any).type === "web_search") {
			return {
				type: "web_search" as any,
				...((tool as any).search_context_size ? { search_context_size: (tool as any).search_context_size } : {}),
				...((tool as any).filters ? { filters: (tool as any).filters } : {}),
				...((tool as any).user_location ? { user_location: (tool as any).user_location } : {}),
				...((tool as any).external_web_access !== undefined
					? { external_web_access: (tool as any).external_web_access }
					: {}),
			}
		}
		return tool
	})

	// Cast to any to support web_search tool type which is not yet in the official OpenAI SDK types
	const finalTools = mappedTools as any[]

	return {
		tools: finalTools,
		tool_choice: "auto" as ChatCompletionToolChoiceOption,
		parallel_tool_calls: enableParallelToolCalls,
	}
}
