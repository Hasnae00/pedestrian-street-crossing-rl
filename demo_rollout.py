import matplotlib.pyplot as plt
from env import EnvParams, Scenario, PedestrianCrossingEnv
from visualize import visualize_step
import time

params = EnvParams()
scenario = Scenario(lambda_lane1=0.30, lambda_lane2=0.15)
env = PedestrianCrossingEnv(params, scenario, seed=1)

state = env.reset()
plt.figure(figsize=(10, 3))

while True:
    tta1, tta2, ped_stage, stage_prog = state
    ped_stage = int(ped_stage)

    # Simple human-like demo policy:
    # - At CURB (stage 0): only decide based on lane 1 safety
    # - At MEDIAN (stage 2): only decide based on lane 2 safety
    # - While crossing (stages 1 and 3): action doesn't matter (no-op)
    if ped_stage == 0:
        # Decide to cross lane 1
        if len(env.cars[0]) > 0 and tta1 > 3.0:
            action = 1  # GO
        else:
            action = 0  # WAIT

    elif ped_stage == 2:
        # Decide to cross lane 2
        if len(env.cars[1]) > 0 and tta2 > 3.0:
            action = 1  # GO
        else:
            action = 0  # WAIT

    else:
        # Crossing stage -> no decision needed
        action = 0

    state, reward, done, info = env.step(action)

    # render
    env_data = env.render_data()
    visualize_step(env_data, spawn_distance=params.spawn_distance_m)

    if done:
        time.sleep(0.5)
        break

    time.sleep(params.dt)

plt.show()
