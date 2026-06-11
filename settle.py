#!/usr/bin/env python3
"""
Settlement + self-audit entrypoint.

  python settle.py            -> score any newly finished matches against all baselines
  python settle.py --audit    -> also send the running self-audit to Telegram (weekly)
"""
import sys

from agent import calibration, telegram


def main() -> None:
    n = calibration.settle()
    print(f"[settle] scored {n} newly finished match(es)")
    if "--audit" in sys.argv:
        telegram.send_message(calibration.audit_text())
        print("[audit] sent")


if __name__ == "__main__":
    main()
