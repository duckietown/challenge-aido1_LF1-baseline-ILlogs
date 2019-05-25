from env import launch_env
from graph_utils import load_graph
import tensorflow as tf
from cnn_predictions import fun_img_preprocessing

# configuration zone
# yes, remember the simulator give us an outrageously large image
# we preprocessed in the logs, but here we rely on the preprocessing step in the model
OBSERVATIONS_SHAPE = (None, 480, 640, 3)
ACTIONS_SHAPE = (None, 2)
SEED = 1234
STORAGE_LOCATION = "trained_models/behavioral_cloning"
EPISODES = 10
STEPS = 650

env = launch_env()


def compute_action(observation):


        return action


observation = env.reset()

# we can use the gym reward to get an idea of the performance of our model
cumulative_reward = 0.0

frozen_model_filename = "frozen_graph.pb"

# We use our "load_graph" function
graph = load_graph(frozen_model_filename)

# To check which operations your network is using
# uncomment the following commands:
# We can verify that we can access the list of operations in the graph
# for op in graph.get_operations():
#     print(op.name)

# We access the input and output nodes
x = graph.get_tensor_by_name('prefix/x:0')
y = graph.get_tensor_by_name('prefix/ConvNet/fc_layer_2/BiasAdd:0')
# We launch a Session
with tf.Session(graph=graph) as sess:

    for episode in range(0, EPISODES):
        for steps in range(0, STEPS):
            # Additionally img is converted to greyscale
            observation = fun_img_preprocessing(observation, 48, 96)
            # this outputs omega, the desired angular velocity
            action = sess.run(y, feed_dict={
                x: observation
            })
            action = [action[0, 0], action[0, 1]]
            # action = compute_action(observation)
            observation, reward, done, info = env.step(action)
            cumulative_reward += reward
            if done:
                env.reset()
            env.render()
        # we reset after each episode, or not, this really depends on you
        env.reset()

print('total reward: {}, mean reward: {}'.format(cumulative_reward, cumulative_reward // EPISODES))
# didn't look good, ah? Well, that's where you come into the stage... good luck!

env.close()