import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from training.gmn_pettingzoo import GMNMultiAgentEnv, EVENT_CODE_MAP

env = GMNMultiAgentEnv(scenario="academy_3_vs_1_with_keeper", auto_start_bridge=False, port=5157,
                       enable_reward_shaping=True, shot_clock_truncates=False, shot_clock_t_max=600)
obs, infos = env.reset(seed=9001)
print("RESET agents:", env.agents)
print("RESET info keys:", sorted(infos[env.agents[0]].keys()))

owner = None
for i in range(200):
    live = list(env.agents)
    if not live:
        break
    obs, r, t, tr, infos = env.step({a: 5 for a in live})
    live = list(env.agents)
    if not live:
        break
    for a in live:
        m = obs[a].get("action_mask") if isinstance(obs[a], dict) else None
        if m is not None and len(m) > 11 and m[11] == 1:
            owner = a
            break
    if owner:
        break
print("owner:", owner, "after acquisition loop")

act = {a: 5 for a in list(env.agents)}
if owner:
    act[owner] = 11
for k in range(8):
    live = list(env.agents)
    if not live:
        print("episode already ended before tick", k)
        break
    probe = live[0]
    obs, r, t, tr, infos = env.step({a: act.get(a, 5) for a in live})
    after = list(env.agents)
    info = infos.get(probe, {})
    ec = info.get("eventCode", 0)
    ecn = EVENT_CODE_MAP[ec] if 0 <= ec < len(EVENT_CODE_MAP) else "?"
    print("--- tick", k, "---")
    print("  before:", live, "after:", after)
    print("  rewards:", r)
    print("  terms:", t, "truncs:", tr)
    print("  eventCode:", ec, ecn)
    print("  step_events:", info.get("step_events"))
    print("  info keys:", sorted(info.keys()))
    print("  reason fields:", {x: info.get(x) for x in ("termination_reason", "terminated", "truncated", "done_reason", "scenarioStatus", "episode_stats", "scenario_result", "reason") if x in info})
    if not after:
        print("  EPISODE ENDED after this step")
        break
    act = {a: 5 for a in after}
env.close()
