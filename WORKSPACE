# -*- mode: python -*-
# vi: set ft=python :

workspace(name = "c3")

DRAKE_COMMIT = "v1.35.0"

DRAKE_CHECKSUM = "f19ee360656c87db8b0688c676c9f2ab2ae71ea08691432979b32d71c236e768"
# Before changing the COMMIT, temporarily uncomment the next line so that Bazel
# displays the suggested new value for the CHECKSUM.
# DRAKE_CHECKSUM = "0" * 64

# Maybe download Drake.
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

http_archive(
    name = "drake",
    sha256 = DRAKE_CHECKSUM,
    strip_prefix = "drake-{}".format(DRAKE_COMMIT.strip("v")),
    urls = [x.format(DRAKE_COMMIT) for x in [
        "https://github.com/RobotLocomotion/drake/archive/{}.tar.gz",
    ]],
)

# Reference external software libraries and tools per Drake's defaults.  Some
# software will come from the host system (Ubuntu or macOS); other software
# will be downloaded in source or binary form from github or other sites.
load("@drake//tools/workspace:default.bzl", "add_default_workspace")

add_default_workspace()

load("@rules_python//python:repositories.bzl", "py_repositories")

py_repositories()

# Hedron's Compile Commands Extractor for Bazel
# https://github.com/hedronvision/bazel-compile-commands-extractor
http_archive(
    name = "hedron_compile_commands",
    strip_prefix = "bazel-compile-commands-extractor-f5fbd4cee671d8d908f37c83abaf70fba5928fc7",

    # Replace the commit hash (0e990032f3c5a866e72615cf67e5ce22186dcb97) in both places (below) with the latest (https://github.com/hedronvision/bazel-compile-commands-extractor/commits/main), rather than using the stale one here.
    # Even better, set up Renovate and let it do the work for you (see "Suggestion: Updates" in the README).
    url = "https://github.com/mikael-s-persson/bazel-compile-commands-extractor/archive/f5fbd4cee671d8d908f37c83abaf70fba5928fc7.tar.gz",
    # When you first run this tool, it'll recommend a sha256 hash to put here with a message like: "DEBUG: Rule 'hedron_compile_commands' indicated that a canonical reproducible form can be obtained by modifying arguments sha256 = ..."
)

load("@hedron_compile_commands//:workspace_setup.bzl", "hedron_compile_commands_setup")

hedron_compile_commands_setup()

load("@hedron_compile_commands//:workspace_setup_transitive.bzl", "hedron_compile_commands_setup_transitive")

hedron_compile_commands_setup_transitive()

load("@hedron_compile_commands//:workspace_setup_transitive_transitive.bzl", "hedron_compile_commands_setup_transitive_transitive")

hedron_compile_commands_setup_transitive_transitive()

load("@hedron_compile_commands//:workspace_setup_transitive_transitive_transitive.bzl", "hedron_compile_commands_setup_transitive_transitive_transitive")

hedron_compile_commands_setup_transitive_transitive_transitive()
