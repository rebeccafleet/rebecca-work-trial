import os, csv, json, argparse, uuid
from datetime import datetime, timedelta, date
from zoneinfo import ZoneInfo
from typing import Optional

import yaml
import numpy as np
import psycopg2


def weighted_choice(rng: np.random.Generator, weights_map: dict) -> str:
    """Pick a key from a dict based on its value as a probability weight."""
    keys = list(weights_map.keys())
    p = np.array(list(weights_map.values()), dtype=float)
    p = p / p.sum()
    return str(rng.choice(keys, p=p))


def weighted_choice_from_list(rng: np.random.Generator, items: list, weight_key: str = "weight") -> dict:
    """Pick an item from a list of dicts using a weight key."""
    weights = np.array([item.get(weight_key, 1.0) for item in items], dtype=float)
    weights = weights / weights.sum()
    idx = rng.choice(len(items), p=weights)
    return items[idx]


def sample_time_of_day(rng: np.random.Generator, pattern: str = "work_hours") -> int:
    """
    Sample a second-of-day based on activity pattern.
    
    Patterns:
      - work_hours: Peak around 2 PM, quiet at night
      - evening_peak: Peak around 8 PM
      - always_on: Uniform distribution
    """
    if pattern == "always_on":
        return int(rng.integers(0, 86400))
    
    if pattern == "evening_peak":
        # 7 pm
        mean_seconds = 19 * 3600
        std_seconds = 3 * 3600
    else:
        # work_hours default here, 1 PM peak
        mean_seconds = 13 * 3600
        std_seconds = 4 * 3600
    
    while True:
        sample = rng.normal(mean_seconds, std_seconds)
        sample = sample % 86400
        second = int(sample)
        
        # Reject samples between 2 AM - 6 AM with 80% probability
        hour = second // 3600
        if 2 <= hour < 6 and rng.random() < 0.8:
            continue
        
        return second


def generate_status_code(rng: np.random.Generator, error_rate: float) -> int:
    """Generate an HTTP status code based on error rate."""
    if rng.random() < error_rate:
        # Error codes weighted distribution
        error_codes = {500: 0.5, 502: 0.2, 503: 0.15, 504: 0.1, 400: 0.05}
        return int(weighted_choice(rng, error_codes))
    return 200


def apply_latency_jitter(rng: np.random.Generator, base_latency: float, multiplier: float = 1.0) -> int:
    """Apply realistic jitter to base latency using gamma distribution."""
    # Gamma shape=2 gives right-skewed distribution like real latencies
    jittered = rng.gamma(2.0, base_latency * multiplier / 2.0)
    return int(max(1, jittered))


class ApplicationLogGenerator:
    """Generates realistic application logs based on consumer.yml and personas.yml."""
    
    def __init__(self, consumer_path: str, personas_path: str):
        with open(consumer_path, "r") as f:
            self.consumer = yaml.safe_load(f)
        with open(personas_path, "r") as f:
            self.personas_cfg = yaml.safe_load(f)
        
        self.tz = ZoneInfo(self.consumer["runtime"]["timezone"])
        self.start = datetime.fromisoformat(self.consumer["runtime"]["start"])
        if self.start.tzinfo is None:
            self.start = self.start.replace(tzinfo=self.tz)
        
        self.duration_days = self.consumer["runtime"]["duration_days"]
        self.tick_seconds = self.consumer["runtime"].get("tick_seconds", 60)
        self.seed = int(self.consumer["rng_seed"])
        self.rng = np.random.default_rng(self.seed)
        
        self.out_dir = self.consumer["output"]["out_dir"]
        os.makedirs(self.out_dir, exist_ok=True)
        
        # Build route lookup by stage
        self.routes_by_stage = self._build_routes_by_stage()
        self.all_routes = self._build_all_routes()
        
        # Build persona lookup
        self.personas = {p["id"]: p for p in self.personas_cfg["personas"]}
        self.persona_defaults = self.personas_cfg.get("defaults", {})
        
        # Build region lookup
        self.regions = self.consumer["topology"]["regions"]
        
        # Journey config
        self.stages = self.consumer["journeys"]["stages"]
        self.transitions = self.consumer["journeys"]["transitions"]
        self.steps_per_stage = self.consumer["journeys"]["steps_per_stage"]
        
        # Feature flags
        self.feature_flags = self.consumer.get("feature_flags", [])
        
    def _build_routes_by_stage(self) -> dict:
        """Group routes by their journey stage."""
        routes_by_stage = {}
        for service in self.consumer["system"]["services"]:
            for route in service["routes"]:
                stage = route.get("stage", "browse")
                if stage not in routes_by_stage:
                    routes_by_stage[stage] = []
                routes_by_stage[stage].append({
                    **route,
                    "service": service["name"]
                })
        return routes_by_stage
    
    def _build_all_routes(self) -> list:
        """Flatten all routes with service info."""
        all_routes = []
        for service in self.consumer["system"]["services"]:
            for route in service["routes"]:
                all_routes.append({
                    **route,
                    "service": service["name"]
                })
        return all_routes
    
    def _get_persona_attr(self, persona_id: str, category: str, key: str):
        """Get a persona attribute with fallback to defaults."""
        persona = self.personas.get(persona_id, {})
        cat_data = persona.get(category, {})
        if key in cat_data:
            return cat_data[key]
        return self.persona_defaults.get(category, {}).get(key)
    
    def _pick_region(self) -> dict:
        """Select a region based on weights."""
        return weighted_choice_from_list(self.rng, self.regions, "weight")
    
    def _pick_persona(self) -> str:
        """Select a persona based on share_of_users."""
        personas_list = self.personas_cfg["personas"]
        return weighted_choice_from_list(self.rng, personas_list, "share_of_users")["id"]
    
    def _check_feature_flag(self, user_id: int, flag_key: str) -> bool:
        """Check if a feature flag is enabled for a user."""
        for flag in self.feature_flags:
            if flag["key"] == flag_key:
                # Simple user-based rollout
                user_hash = hash((flag_key, user_id)) % 100 / 100.0
                return user_hash < flag.get("rollout", 0)
        return False
    
    def _next_stage(self, current_stage: str) -> Optional[str]:
        """Determine the next stage based on transition probabilities."""
        if current_stage not in self.transitions:
            return None
        
        trans = self.transitions[current_stage]
        next_stage = weighted_choice(self.rng, trans)
        
        if next_stage == "end":
            return None
        return next_stage
    
    def generate_users(self) -> list:
        """Generate user records."""
        user_count = self.consumer["users"]["count"]
        countries = self.consumer["users"]["countries"]
        internal_rate = self.consumer["users"].get("internal_user_rate", 0.02)
        
        users = []
        for uid in range(1, user_count + 1):
            persona_id = self._pick_persona()
            device_mix = self._get_persona_attr(persona_id, "identity", "device_mix") or {"web": 0.7, "mobile": 0.3}
            
            created = self.start - timedelta(days=int(self.rng.integers(0, 90)))
            
            users.append({
                "user_id": uid,
                "created_at": created.isoformat(),
                "country": weighted_choice(self.rng, countries),
                "persona": persona_id,
                "is_internal": bool(self.rng.random() < internal_rate),
                "device_primary": weighted_choice(self.rng, device_mix),
            })
        
        return users
    
    def generate_session(self, user: dict, session_start: datetime) -> list:
        """Generate a user session as a sequence of log entries following a journey."""
        logs = []
        session_id = str(uuid.uuid4())[:8]
        persona_id = user["persona"]
        user_id = user["user_id"]
        
        # Get persona-specific settings
        auth_rate = self._get_persona_attr(persona_id, "identity", "auth_rate") or 0.95
        device_mix = self._get_persona_attr(persona_id, "identity", "device_mix") or {"web": 0.7, "mobile": 0.3}
        timeout_mult = self._get_persona_attr(persona_id, "quality", "timeout_multiplier") or 1.0
        retry_mult = self._get_persona_attr(persona_id, "quality", "retry_multiplier") or 1.0
        malformed_rate = self._get_persona_attr(persona_id, "quality", "malformed_log_rate") or 0.0005
        
        # Session starts at browse stage
        current_stage = "browse"
        current_time = session_start
        region = self._pick_region()
        device = weighted_choice(self.rng, device_mix)
        
        # Is this an authenticated session?
        is_authenticated = self.rng.random() < auth_rate
        effective_user_id = user_id if is_authenticated else None
        
        while current_stage:
            # How many steps in this stage?
            stage_cfg = self.steps_per_stage.get(current_stage, {"min": 1, "max": 3})
            num_steps = int(self.rng.integers(stage_cfg["min"], stage_cfg["max"] + 1))
            
            routes_for_stage = self.routes_by_stage.get(current_stage, [])
            if not routes_for_stage:
                # No routes for this stage, pick from all routes
                routes_for_stage = self.all_routes
            
            for _ in range(num_steps):
                route = weighted_choice_from_list(self.rng, routes_for_stage, "weight")
                
                # Calculate latency with persona modifier
                base_latency = route["base_latency_ms"]
                latency = apply_latency_jitter(self.rng, base_latency, timeout_mult)
                
                # Check feature flags for route overrides
                for flag in self.feature_flags:
                    if self._check_feature_flag(user_id, flag["key"]):
                        overrides = flag.get("effects", {}).get("route_overrides", {})
                        route_key = f"{route['service']}.{route['id']}"
                        if route_key in overrides:
                            lat_mult = overrides[route_key].get("latency_multiplier", 1.0)
                            latency = int(latency * lat_mult)
                
                # Calculate error rate
                error_rate = route["base_error_rate"]
                status_code = generate_status_code(self.rng, error_rate)
                
                # Should we retry on error?
                attempts = 1
                if status_code >= 500 and self.rng.random() < 0.3 * retry_mult:
                    attempts = int(self.rng.integers(2, 4))
                
                for attempt in range(attempts):
                    # Generate malformed log entry sometimes
                    is_malformed = self.rng.random() < malformed_rate
                    
                    log_entry = {
                        "request_id": str(uuid.uuid4()),
                        "timestamp": current_time.isoformat(),
                        "user_id": effective_user_id,
                        "session_id": session_id,
                        "service": route["service"],
                        "route_id": route["id"],
                        "method": route["method"],
                        "path": route["path"],
                        "status_code": status_code if not is_malformed else None,
                        "latency_ms": latency if not is_malformed else -1,
                        "region": region["id"],
                        "device": device,
                        "persona": persona_id,
                        "stage": current_stage,
                        "attempt": attempt + 1,
                        "is_malformed": is_malformed,
                    }
                    logs.append(log_entry)
                    
                    # Small delay between retries
                    if attempt < attempts - 1:
                        current_time += timedelta(milliseconds=int(self.rng.integers(100, 2000)))
                        # Retry might succeed
                        status_code = generate_status_code(self.rng, error_rate * 0.5)
                
                # Advance time between requests in session
                current_time += timedelta(seconds=int(self.rng.integers(1, 30)))
            
            # Transition to next stage
            current_stage = self._next_stage(current_stage)
        
        return logs
    
    def generate(self):
        """Main generation loop."""
        print(f"Generating application logs...")
        print(f"  Duration: {self.duration_days} days starting {self.start.date()}")
        print(f"  Users: {self.consumer['users']['count']}")
        
        # Generate users
        users = self.generate_users()
        users_csv = os.path.join(self.out_dir, "dim_users.csv")
        with open(users_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["user_id", "created_at", "country", "persona", "is_internal", "device_primary"])
            w.writeheader()
            w.writerows(users)
        print(f"  Generated {len(users)} users -> {users_csv}")
        
        # Generate application logs
        logs_csv = os.path.join(self.out_dir, "application_logs.csv")
        log_fields = [
            "request_id", "timestamp", "user_id", "session_id", "service", 
            "route_id", "method", "path", "status_code", "latency_ms",
            "region", "device", "persona", "stage", "attempt", "is_malformed"
        ]
        
        total_logs = 0
        users_by_id = {u["user_id"]: u for u in users}
        
        with open(logs_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=log_fields)
            w.writeheader()
            
            # For each day, generate sessions
            for day_offset in range(self.duration_days):
                day_start = self.start + timedelta(days=day_offset)
                day_logs = []
                
                # Each user has a chance to have sessions based on persona
                for user in users:
                    persona_id = user["persona"]
                    rate_mult = self._get_persona_attr(persona_id, "activity", "request_rate_multiplier") or 1.0
                    pattern = self._get_persona_attr(persona_id, "activity", "active_hours_pattern") or "work_hours"
                    burstiness = self._get_persona_attr(persona_id, "activity", "burstiness") or 0.3
                    
                    # Determine number of sessions for this user today
                    # Base: 0-2 sessions, modified by rate multiplier
                    base_sessions = self.rng.poisson(0.5 * rate_mult)
                    
                    # Add bursty behavior
                    if self.rng.random() < burstiness:
                        base_sessions += int(self.rng.integers(1, 4))
                    
                    for _ in range(min(base_sessions, 10)):  # Cap at 10 sessions/day
                        second_of_day = sample_time_of_day(self.rng, pattern)
                        session_start = day_start + timedelta(seconds=second_of_day)
                        
                        session_logs = self.generate_session(user, session_start)
                        day_logs.extend(session_logs)
                
                # Sort by timestamp and write
                day_logs.sort(key=lambda x: x["timestamp"])
                w.writerows(day_logs)
                total_logs += len(day_logs)
                print(f"    Day {day_offset + 1}: {len(day_logs)} log entries")
        
        print(f"  Generated {total_logs} log entries -> {logs_csv}")
        
        # Write manifest
        manifest = {
            "users_csv": users_csv,
            "logs_csv": logs_csv,
            "generated_at": datetime.utcnow().isoformat(),
            "config": {
                "duration_days": self.duration_days,
                "user_count": len(users),
                "total_logs": total_logs,
                "rng_seed": self.seed,
            }
        }
        manifest_path = os.path.join(self.out_dir, "manifest.json")
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        
        print(f"Generated manifest -> {manifest_path}")


def generate(consumer_path: str, personas_path: str) -> None:
    """Generate application logs from consumer.yml and personas.yml."""
    gen = ApplicationLogGenerator(consumer_path, personas_path)
    gen.generate()


def init_db():
    """Initialize database schema."""
    conn = psycopg2.connect(os.environ["DATABASE_URL"])
    cur = conn.cursor()
    cur.execute("""
      CREATE TABLE IF NOT EXISTS dim_users (
        user_id BIGINT PRIMARY KEY,
        created_at TIMESTAMPTZ NOT NULL,
        country TEXT NOT NULL,
        persona TEXT NOT NULL,
        is_internal BOOLEAN NOT NULL,
        device_primary TEXT NOT NULL
      );
    """)
    cur.execute("""
      CREATE TABLE IF NOT EXISTS application_logs (
        request_id UUID PRIMARY KEY,
        timestamp TIMESTAMPTZ NOT NULL,
        user_id BIGINT REFERENCES dim_users(user_id),
        session_id TEXT NOT NULL,
        service TEXT NOT NULL,
        route_id TEXT NOT NULL,
        method TEXT NOT NULL,
        path TEXT NOT NULL,
        status_code INT,
        latency_ms INT NOT NULL,
        region TEXT NOT NULL,
        device TEXT NOT NULL,
        persona TEXT NOT NULL,
        stage TEXT NOT NULL,
        attempt INT NOT NULL,
        is_malformed BOOLEAN NOT NULL
      );
    """)
    conn.commit()
    conn.close()
    print("Initialized tables.")


def load(consumer_path: str):
    """Load generated CSVs into Postgres."""
    with open(consumer_path, "r") as f:
        cfg = yaml.safe_load(f)
    out_dir = cfg["output"]["out_dir"]

    conn = psycopg2.connect(os.environ["DATABASE_URL"])
    cur = conn.cursor()

    with open(os.path.join(out_dir, "dim_users.csv"), "r") as f:
        cur.copy_expert("COPY dim_users FROM STDIN WITH (FORMAT csv, HEADER true)", f)

    with open(os.path.join(out_dir, "application_logs.csv"), "r") as f:
        cur.copy_expert("COPY application_logs FROM STDIN WITH (FORMAT csv, HEADER true)", f)

    conn.commit()
    conn.close()
    print("Loaded CSVs into Postgres.")


def main():
    p = argparse.ArgumentParser(description="Generate application logs from consumer/personas manifests")
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="Generate application logs to CSV")
    g.add_argument("--consumer", required=True, help="Path to consumer.yml")
    g.add_argument("--personas", required=True, help="Path to personas.yml")

    sub.add_parser("init-db", help="Initialize database schema")

    l = sub.add_parser("load", help="Load CSVs into Postgres")
    l.add_argument("--consumer", required=True, help="Path to consumer.yml")

    args = p.parse_args()

    if args.cmd == "generate":
        generate(args.consumer, args.personas)
    elif args.cmd == "init-db":
        init_db()
    elif args.cmd == "load":
        load(args.consumer)


if __name__ == "__main__":
    main()
