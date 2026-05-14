# IMPORTANT: `npm run postpublish` to update this file after publishing a new version of the package
class Dirac < Formula
  desc "Autonomous coding agent CLI - capable of creating/editing files, running commands, and more"
  homepage "https://dirac.run"
  url "https://registry.npmjs.org/dirac-cli/-/dirac-cli-0.3.43.tgz" # GET from https://registry.npmjs.org/dirac-cli/latest tarball URL
  sha256 "89808bcbccde302fb5fcbeba60f48cf9653628d0faeb10786dd2224e397182bd"
  license :cannot_represent

  depends_on "node@20"
  depends_on "ripgrep"

  def install
    system "npm", "install", *std_npm_args(prefix: false)
    bin.install_symlink Dir["#{libexec}/bin/*"]
  end

  test do
    # Test that the binary exists and is executable
    assert_match version.to_s, shell_output("#{bin}/dirac --version")
  end
end
