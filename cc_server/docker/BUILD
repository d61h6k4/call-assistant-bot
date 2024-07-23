
load("@rules_oci//oci:defs.bzl", "oci_image", "oci_push", "oci_tarball")
load("@rules_pkg//pkg:tar.bzl", "pkg_tar")



pkg_tar(
    name = "cc_server_tar",
    srcs = [
        "//cc_server",
    ],
    include_runfiles = True,
    strip_prefix = ".",
)

oci_image(
    name = "image",
    base = "@distroless_python3",
    entrypoint = ["/cc_server/cc_server"],
    tars = [
        ":cc_server_tar",
    ],
)

# Build: bazel run //cc_server/docker:tarball
# Run: docker run --rm -e AUTH_TOKEN=token aikit/cc_server:latest
oci_tarball(
    name = "tarball",
    image = ":image",
    repo_tags = [
        "aikit/cc_server:latest",
    ],
)


oci_push(
    name = "push",
    image = ":image",
    remote_tags = ["latest"],
    repository = "europe-west1-b-docker.pkg.dev/ai-call-bot-424111/cc_server/cc_server",
)
