"""把主问题的 x/y/m/n/z 写成 JSON，并按 TAZ 编号列出部署。"""

from __future__ import annotations

from settings import DATE_TYPES


def json_decision(decisions: dict, zone_count: int, hub_count: int, date_count: int) -> dict:
    return {
        "x": {str(hub): int(decisions["x"][hub]) for hub in range(hub_count)},
        "m": {str(hub): int(decisions["m"][hub]) for hub in range(hub_count)},
        "n": {str(zone): int(decisions["n"][zone]) for zone in range(zone_count)},
        "y": {
            f"{zone},{hub}": int(decisions["y"][zone, hub])
            for zone in range(zone_count)
            for hub in range(hub_count)
        },
        "z": {
            f"{zone},{hub},{date_type}": int(decisions["z"][zone, hub, date_type])
            for zone in range(zone_count)
            for hub in range(hub_count)
            for date_type in range(date_count)
        },
    }


def labeled_deployment(zone_ids, decision: dict) -> dict:
    count = len(zone_ids)
    hubs = []
    for index, taz_id in enumerate(zone_ids):
        if int(decision["x"][str(index)]) != 1:
            continue
        hubs.append({
            "index": index,
            "taz_id": int(taz_id),
            "mobile_chargers": int(decision["m"][str(index)]),
        })
    zones = []
    for zone, taz_id in enumerate(zone_ids):
        assigned = [
            int(zone_ids[hub])
            for hub in range(count)
            if int(decision["y"][f"{zone},{hub}"]) == 1
        ]
        dispatch = {}
        for date_index, date_name in enumerate(DATE_TYPES):
            dispatch[date_name] = [
                {
                    "hub_taz_id": int(zone_ids[hub]),
                    "mobile_chargers": int(decision["z"][f"{zone},{hub},{date_index}"]),
                }
                for hub in range(count)
                if int(decision["z"][f"{zone},{hub},{date_index}"]) > 0
            ]
        zones.append({
            "index": zone,
            "taz_id": int(taz_id),
            "fixed_chargers": int(decision["n"][str(zone)]),
            "hub_taz_id": None if not assigned else assigned[0],
            "dispatch": dispatch,
        })
    return {"hubs": hubs, "zones": zones}


def attach_deployment(payload: dict, zone_ids, decisions: dict) -> None:
    zone_count = len(zone_ids)
    payload["decision"] = json_decision(
        decisions, zone_count, zone_count, len(DATE_TYPES),
    )
    payload["deployment"] = labeled_deployment(zone_ids, payload["decision"])
