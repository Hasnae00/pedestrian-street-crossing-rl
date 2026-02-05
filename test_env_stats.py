import numpy as np
from env import EnvParams, Scenario, PedestrianCrossingEnv

def run(env, episodes=200, margin=0.8):
    succ = coll = tout = 0
    for _ in range(episodes):
        s = env.reset()
        done = False
        while not done:
            tta1, tta2, stage, prog = s
            stage = int(stage)

            if stage == env.STAGE_CURB:
                a = env.ACTION_GO if (tta1 > env.params.t_lane_s + margin) else env.ACTION_WAIT
            elif stage == env.STAGE_MEDIAN:
                a = env.ACTION_GO if (tta2 > env.params.t_lane_s + margin) else env.ACTION_WAIT
            else:
                a = env.ACTION_WAIT

            s, r, done, info = env.step(a)

        ev = info["terminal_event"]
        if ev == "success":
            succ += 1
        elif ev == "collision":
            coll += 1
        else:
            tout += 1

    return succ, coll, tout

def main():
    params = EnvParams()
    scenario = Scenario(lambda_lane1=0.65, lambda_lane2=0.35)
    env = PedestrianCrossingEnv(params, scenario, seed=0)

    succ, coll, tout = run(env, episodes=300, margin=0.8)
    print(f"heuristic: succ={succ/300:.2%}, coll={coll/300:.2%}, tout={tout/300:.2%}, margin= {0.8}")

if __name__ == "__main__":
    main()
