#!/usr/bin/env python
"""Insert synthetic demo participants so the API and dashboard have data.

These rows are generated profiles, not humans: each participant carries
`source="synthetic"` and the consent rows exist so the consent gate can be
exercised. Do not mix them with pilot participants.

    python scripts/seed_demo.py --n 100
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.core import consent as consent_mod  # noqa: E402
from backend.core.security import hash_token, new_token  # noqa: E402
from backend.db.models import Participant, Profile  # noqa: E402
from backend.db.session import get_sessionmaker, init_db  # noqa: E402
from backend.experiments.simulation import generate_profiles  # noqa: E402
from backend.features.schema import FEATURE_SCHEMA_VERSION  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=20261018)
    args = ap.parse_args()

    init_db()
    session = get_sessionmaker()()
    profiles = generate_profiles(args.n, seed=args.seed)

    created = 0
    for profile in profiles:
        features = {k: v for k, v in profile.items()
                    if not k.startswith("_") and k not in
                    ("participant_id", "profile_id", "version",
                     "feature_schema_version", "updated_at")}
        token = new_token()
        participant = Participant(source="synthetic", token_hash=hash_token(token))
        session.add(participant)
        session.flush()
        consent_mod.record_consent(session, participant.id, True)
        session.add(Profile(participant_id=participant.id, version=1,
                            feature_schema_version=FEATURE_SCHEMA_VERSION,
                            features=features))
        created += 1
    session.commit()
    session.close()
    print(f"seeded {created} synthetic participants (source='synthetic', seed {args.seed})")
    print("these are generated profiles, not research participants")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
