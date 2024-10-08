load("@aspect_bazel_lib//lib:tar.bzl", "tar")
load("@container_structure_test//:defs.bzl", "container_structure_test")
load("@rules_distroless//apt:defs.bzl", "dpkg_status")
load("@rules_distroless//distroless:defs.bzl", "cacerts", "group", "home", "passwd")
load("@rules_oci//oci:defs.bzl", "oci_image", "oci_push", "oci_tarball")
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
            "uid": 1000,
            "gid": 1000,
            "home": "/home/nonroot",
            "shell": "/bin/bash",
            "username": "nonroot",
        },
        {
            "uid": 106,
            "gid": 110,
            "username": "messagebus",
            "home": "/var/run/dbus",
            "shell": "/bin/false",
        },
        {
            "uid": 107,
            "gid": 107,
            "username": "pulse",
            "home": "/var/run/pulse",
            "shell": "/bin/false",
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
            "name": "messagebus",
            "gid": 110,
            "users": [
                "root",
                "nonroot",
                "messagebus",
            ],
        },
        {
            "name": "pulse",
            "gid": 107,
            "users": [
                "root",
                "pulse",
                "nonroot",
            ],
        },
        {
            "name": "pulse-access",
            "gid": 108,
            "users": [
                "root",
                "pulse",
                "nonroot",
            ],
        },
        {
            "name": "sudo",
            "gid": 112,
            "users": [
                "nonroot",
            ],
        },
        {
            "name": "audio",
            "gid": 1,
            "users": [
                "nonroot",
                "root",
                "pulse",
            ],
        },
        {
            "name": "video",
            "gid": 2,
            "users": [
                "nonroot",
                "root",
            ],
        },
    ],
)

home(
    name = "home",
    dirs = [
        {
            "home": "/home/nonroot",
            "uid": 1000,
            "gid": 1000,
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
    "@bookworm//bash",
    "@bookworm//coreutils",
    "@bookworm//xvfb",
    "@bookworm//fluxbox",
    "@bookworm//x11-apps",
    "@bookworm//pulseaudio",
    "@bookworm//chromium-driver",
    "@bookworm//sudo",
    "@bookworm//python3",
    "@bookworm//dbus",
    "@bookworm//dbus-x11",
    "@bookworm//eterm",
    "@bookworm//gfortran",
]

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
    name = "image",
    base = "@base",
    entrypoint = ["/meeting_bot/meeting_bot"],
    env = {
        "DISPLAY": ":1.0",
        # tells any program you run where to find a nonroot-specific directory
        # in which it can store small temporary files.
        "XDG_RUNTIME_DIR": "/home/nonroot/0",
        # Pulseaudio ignores XDG_RUNTIME_DIR
        "PULSE_SERVER": "/var/run/pulse/native",
        "DBUS_STARTER_BUS_TYPE": "session",
        "DBUS_SESSION_BUS_ADDRESS": "unix:path=/run/dbus/system_bus_socket",
        "TEMP": "/home/nonroot",
        "OMP_NUM_THREADS": 4,
    },
    tars = [
        ":sh",
        ":passwd",
        ":group",
        ":home",
        ":packages_tar",
        ":meeting_bot_tar",
    ],
    user = "root",
)

# Build: bazel run //docker:tarball
# Run: docker run --rm aikit/meetingbot:latest \
#                      --gmeet_link=https://meet.google.com/rof-kvpv-yue
oci_tarball(
    name = "tarball",
    image = ":image",
    repo_tags = [
        "aikit/meetingbot:latest",
    ],
)

container_structure_test(
    name = "test",
    configs = select({
        "@platforms//cpu:arm64": ["test_linux_arm64.yaml"],
        "@platforms//cpu:x86_64": ["test_linux_amd64.yaml"],
    }),
    image = ":image",
)

oci_push(
    name = "push",
    image = ":image",
    remote_tags = ["latest"],
    repository = "europe-west1-b-docker.pkg.dev/ai-call-bot-424111/meeting-bot/meeting-bot",
)
