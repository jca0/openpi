import dataclasses
import enum
import logging
import socket

import tyro

from openpi.policies import policy as _policy
from openpi.policies import policy_config as _policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as _config


class EnvMode(enum.Enum):
    """Supported environments."""

    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"


@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""

    # Training config name (e.g., "pi0_aloha_sim").
    config: str
    # Checkpoint directory (e.g., "checkpoints/pi0_aloha_sim/exp/10000").
    dir: str


@dataclasses.dataclass
class Default:
    """Use the default policy for the given environment."""


@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy script."""

    # Environment to serve the policy for. This is only used when serving default policies.
    env: EnvMode = EnvMode.ALOHA_SIM

    # If provided, will be used in case the "prompt" key is not present in the data, or if the model doesn't have a default
    # prompt.
    default_prompt: str | None = None

    # Port to serve the policy on.
    port: int = 8000
    # Record the policy's behavior for debugging.
    record: bool = False

    # Dynamic prompting: decompose task into subtasks and advance automatically.
    dynamic_prompting: bool = False
    # Calibration: test N decomposition variations and log per-subtask success/failure.
    calibration: bool = False
    # Number of decomposition variations to test during calibration.
    calibration_n_variations: int = 5
    # How often (seconds) to query Gemini for subtask completion.
    check_interval_sec: float = 1.0

    # Specifies how to load the policy. If not provided, the default policy for the environment will be used.
    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)


# Default checkpoints that should be used for each environment.
DEFAULT_CHECKPOINT: dict[EnvMode, Checkpoint] = {
    EnvMode.ALOHA: Checkpoint(
        config="pi05_aloha",
        dir="gs://openpi-assets/checkpoints/pi05_base",
    ),
    EnvMode.ALOHA_SIM: Checkpoint(
        config="pi0_aloha_sim",
        dir="gs://openpi-assets/checkpoints/pi0_aloha_sim",
    ),
    EnvMode.DROID: Checkpoint(
        config="pi05_droid",
        dir="gs://openpi-assets/checkpoints/pi05_droid",
    ),
    EnvMode.LIBERO: Checkpoint(
        config="pi05_libero",
        dir="gs://openpi-assets/checkpoints/pi05_libero",
    ),
}


def create_default_policy(env: EnvMode, *, default_prompt: str | None = None) -> _policy.Policy:
    """Create a default policy for the given environment."""
    if checkpoint := DEFAULT_CHECKPOINT.get(env):
        return _policy_config.create_trained_policy(
            _config.get_config(checkpoint.config), checkpoint.dir, default_prompt=default_prompt
        )
    raise ValueError(f"Unsupported environment mode: {env}")


def create_policy(args: Args) -> _policy.Policy:
    """Create a policy from the given arguments."""
    match args.policy:
        case Checkpoint():
            return _policy_config.create_trained_policy(
                _config.get_config(args.policy.config), args.policy.dir, default_prompt=args.default_prompt
            )
        case Default():
            return create_default_policy(args.env, default_prompt=args.default_prompt)


def main(args: Args) -> None:
    policy = create_policy(args)
    policy_metadata = policy.metadata

    # Record the policy's behavior.
    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")

    # Dynamic prompting / calibration wrappers.
    if args.dynamic_prompting and args.calibration:
        raise ValueError("Cannot use both --dynamic-prompting and --calibration at the same time")

    if args.dynamic_prompting:
        from openpi.dynamic_prompting.policy_wrapper import DynamicPromptingPolicy

        if not args.default_prompt:
            raise ValueError("--default-prompt is required when using --dynamic-prompting")
        logging.info("Dynamic prompting enabled (prompt: %s, check interval: %.1fs)", args.default_prompt, args.check_interval_sec)
        policy = DynamicPromptingPolicy(policy, instruction=args.default_prompt, check_interval_sec=args.check_interval_sec)

    if args.calibration:
        from openpi.dynamic_prompting.policy_wrapper import CalibrationPolicy

        if not args.default_prompt:
            raise ValueError("--default-prompt is required when using --calibration")
        logging.info(
            "Calibration enabled (prompt: %s, variations: %d, check interval: %.1fs)",
            args.default_prompt,
            args.calibration_n_variations,
            args.check_interval_sec,
        )
        policy = CalibrationPolicy(
            policy,
            instruction=args.default_prompt,
            n_variations=args.calibration_n_variations,
            check_interval_sec=args.check_interval_sec,
        )

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    # Suppress noisy loggers
    logging.getLogger("google_genai").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    main(tyro.cli(Args))
