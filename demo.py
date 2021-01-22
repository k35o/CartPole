import gym
import matplotlib.pyplot as plt
from matplotlib import animation
# 環境の生成 
env = gym.make('CartPole-v0')
frames = []
# 環境の初期か
observation = env.reset()
for t in range(1000):
    # 現在の状況を表示させる
    frames.append(env.render("rgb_array"))
    # サンプルの行動をさせる　返り値は左から台車および棒の状態、得られた報酬、ゲーム終了フラグ、詳細情報
    observation, reward, done, info = env.step(env.action_space.sample())
    if done:
        print("Finished after {} timesteps".format(t+1))
        break
# 環境を閉じる
plt.figure()
patch = plt.imshow(frames[0])
plt.axis('off')

def animate(i):
    patch.set_data(frames[i])

anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames),
                                interval=50)
anim.save('demo.gif', "ffmpeg")
env.close()