
"""Global settings loaded from .env file.

Usage:
    from config.settings import settings
    print(settings.API_KEY)
"""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

# Repo root (config/ -> repo root). Anchors .env and the default ArcticDB paths to
# fixed ABSOLUTE locations regardless of the current working directory. Point
# ARCTICDB_DIR / ARCTICDB_SPOT_DIR in .env at an existing store to share it.
_ENGINE_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(_ENGINE_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── Upstox API ──
    UPSTOX_API_KEY: str = ""
    UPSTOX_API_SECRET: str = ""
    UPSTOX_ACCESS_TOKEN: str = ""
    UPSTOX_REDIRECT_URI: str = "http://localhost:8000/callback"
    UPSTOX_BASE_URL: str = "https://api.upstox.com"

    # ── Kite Connect API (Zerodha) ──
    # Kite spot (fetch-kite-equities) and indices (fetch-kite-indices). The daily
    # access token is written by `kite-login --request-token ...` into KITE_ENV_PATH
    # (this repo's .env) and re-read from there by KiteFetcher.
    KITE_API_KEY: str = ""
    KITE_API_SECRET: str = ""
    KITE_ACCESS_TOKEN: str = ""
    KITE_ENV_PATH: Path = _ENGINE_ROOT / ".env"
    KITE_HISTORICAL_RPS: float = 2.5      # stay under the 3/sec hard limit
    KITE_TIMEOUT: int = 30                # SDK default of 7s is too low for big pulls
    KITE_INSTRUMENTS_CACHE_DIR: Path = _ENGINE_ROOT / "data_cache" / "kite_instruments"

    # ── Rate limiting ──
    RATE_LIMIT_PER_SECOND: int = 20       # stay under 25/sec hard limit
    RATE_LIMIT_PER_MINUTE: int = 230      # stay under 250/min
    RATE_LIMIT_PER_30MIN: int = 950       # stay under 1000/30min

    # ── Data storage ──
    DATA_DIR: Path = Path("./data_cache")
    # ArcticDB (LMDB backend). Absolute; override in .env to point at an existing store.
    ARCTICDB_DIR: Path = _ENGINE_ROOT / "data_cache" / "arcticdb"
    # `spot` and `spot_adjusted` live at their OWN root path, separately from
    # every other library, so they can carry a larger LMDB map_size without a
    # second `Arctic()` instance colliding with the main one over the SAME path
    # -- LMDB explicitly does not support opening one path twice in a process
    # (see ArcticStore.__init__). Moved out of ARCTICDB_DIR 2026-09-20 (spot),
    # spot_adjusted following the same day once it also outgrew the default.
    ARCTICDB_SPOT_DIR: Path = _ENGINE_ROOT / "data_cache" / "arcticdb_spot"

    # ── Backtesting defaults ──
    DEFAULT_CAPITAL: float = 1_000_000.0
    DEFAULT_CURRENCY: str = "INR"
    DEFAULT_TIMEZONE: str = "Asia/Kolkata"
    RISK_FREE_RATE: float = 0.065         # ~6.5% India 10Y yield

    # ── Risk defaults ──
    MAX_PORTFOLIO_DRAWDOWN: float = 0.15
    MAX_SINGLE_POSITION_PCT: float = 0.05
    MAX_PORTFOLIO_DELTA: float = 500.0
    MAX_PORTFOLIO_VEGA: float = 50_000.0

    # ── Logging ──
    LOG_LEVEL: str = "INFO"
    LOG_DIR: Path = Path("./logs")

    @property
    def ARCTICDB_URI(self) -> str:
        """LMDB connection URI for the main ArcticDB store (every library except
        `spot` -- see ``ARCTICDB_SPOT_URI``).

        Resolved to an absolute path so every script connects to the same physical
        store regardless of the current working directory.

        Deliberately NOT carrying ``map_size``: ``Arctic(uri)`` applies one
        map_size to EVERY library opened through it, and on this Windows/NTFS
        build opening a library EAGERLY allocates real disk equal to that
        map_size, regardless of actual content (verified 2026-09-20). Bumping
        this to 20GB once tried to grow all 12 libraries at once and ate ~66GB
        of real disk before erroring. Only `spot` genuinely needs headroom
        beyond LMDB's small unconfigured default; it gets that through a
        separate path entirely (below), not a blanket bump here.
        """
        return f"lmdb://{Path(self.ARCTICDB_DIR).resolve()}"

    @property
    def ARCTICDB_SPOT_URI(self) -> str:
        """LMDB connection URI for the `spot` library, carrying a larger
        ``map_size`` (see ``ARCTICDB_SPOT_DIR``'s docstring for why this is a
        physically separate path rather than a second connection over
        ``ARCTICDB_URI``'s path -- LMDB does not support opening one path twice
        in a process, and ArcticDB warns loudly if you try).

        8GB: real content measured 2026-09-20 via an LMDB compact-copy was
        ~2.1GB already committed (15minute+30minute+5minute complete for all 500
        Nifty 500 tickers, `minute` partial) plus ~2.6GB estimated for the
        remaining `minute`-interval backfill -- ~4.7GB total. 8GB leaves
        comfortable headroom for that plus ordinary day-to-day appends without
        needing another bump soon.
        """
        return f"lmdb://{Path(self.ARCTICDB_SPOT_DIR).resolve()}?map_size=8GB"

    def ensure_dirs(self) -> None:
        """Create data and log directories if they don't exist."""
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.ARCTICDB_DIR.mkdir(parents=True, exist_ok=True)
        self.ARCTICDB_SPOT_DIR.mkdir(parents=True, exist_ok=True)
        self.LOG_DIR.mkdir(parents=True, exist_ok=True)


settings = Settings()
