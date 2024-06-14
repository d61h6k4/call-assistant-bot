load("@aspect_bazel_lib//lib:tar.bzl", "tar")
load("@container_structure_test//:defs.bzl", "container_structure_test")
load("@rules_distroless//apt:defs.bzl", "dpkg_status")
load("@rules_distroless//distroless:defs.bzl", "cacerts", "flatten", "group", "passwd")
load("@rules_oci//oci:defs.bzl", "oci_image", "oci_tarball")
load("@rules_pkg//pkg:tar.bzl", "pkg_tar")

passwd(
    name = "passwd",
    entries = [
        {
            "uid": 0,
            "gid": 0,
            "home": "/root",
            "shell": "/bin/bash",
            "username": "root",
        },
        {
            "uid": 100,
            "gid": 65534,
            "home": "/home/pulse",
            "shell": "/usr/sbin/nologin",
            "username": "pulse",
        },
        {
            "uid": 999,
            "gid": 1000,
            "home": "/home/messagebus",
            "shell": "/usr/sbin/nologin",
            "username": "messagebus",
        },
    ],
)

group(
    name = "group",
    entries = [
        {
            "name": "root",
            "gid": 0,
        },
        {
            "name": "pulse",
            "gid": 65534,
        },
        {
            "name": "messagebus",
            "gid": "1000",
        },
        {
            "name": "pulse-access",
            "gid": "1001",
            "users": [
                "root",
            ],
        },
    ],
)

tar(
    name = "sh",
    mtree = [
        # needed as dpkg assumes sh is installed in a typical debian installation.
        "./bin/sh type=link link=/bin/bash",
    ],
)

cacerts(
    name = "cacerts",
    package = select({
        "@platforms//cpu:arm64": "@bookworm//ca-certificates/arm64:data",
        "@platforms//cpu:x86_64": "@bookworm//ca-certificates/amd64:data",
    }),
)

PACKAGES = [
    "@bookworm//ncurses-base",
    "@bookworm//libncurses6",
    "@bookworm//tzdata",
    "@bookworm//bash",
    "@bookworm//coreutils",
    "@bookworm//dpkg",
    "@bookworm//apt",
    "@bookworm//perl",
    "@bookworm//dbus",
    "@bookworm//dbus-x11",
    "@bookworm//xvfb",
    "@bookworm//pulseaudio",
    "@bookworm//grep",
    "@bookworm//chromium",
    "@bookworm//findutils",
    "@bookworm//libfontconfig1",
    "@bookworm//libfreetype6",
    "@bookworm//xfonts-scalable",
    "@bookworm//fonts-liberation",
    "@bookworm//fonts-ipafont-gothic",
]

# Creates /var/lib/dpkg/status with installed package information.
dpkg_status(
    name = "dpkg_status",
    controls = select({
        "@platforms//cpu:arm64": [
            "%s/arm64:control" % package
            for package in PACKAGES
        ],
        "@platforms//cpu:x86_64": [
            "%s/amd64:control" % package
            for package in PACKAGES
        ],
    }),
)

pkg_tar(
    name = "packages_tar",
    deps = select({
        "@platforms//cpu:arm64": [
            "%s/arm64" % package
            for package in PACKAGES
        ],
        "@platforms//cpu:x86_64": [
            "%s/amd64" % package
            for package in PACKAGES
        ],
    }),
)

pkg_tar(
    name = "meeting_bot_tar",
    srcs = [
        "//meeting_bot",
    ],
    include_runfiles = True,
    strip_prefix = ".",
)

oci_image(
    name = "apt",
    base = "@distroless_base",
    entrypoint = ["/meeting_bot/meeting_bot"],
    tars = [
        ":sh",
        ":passwd",
        ":group",
        ":dpkg_status",
        ":cacerts",
        ":packages_tar",
        ":meeting_bot_tar",
    ],
)

oci_tarball(
    name = "tarball",
    image = ":apt",
    repo_tags = [
        "distroless/test:latest",
    ],
)

container_structure_test(
    name = "test",
    configs = select({
        "@platforms//cpu:arm64": ["test_linux_arm64.yaml"],
        "@platforms//cpu:x86_64": ["test_linux_amd64.yaml"],
    }),
    image = ":apt",
)
