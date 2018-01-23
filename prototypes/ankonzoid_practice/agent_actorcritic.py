import random
import numpy as np
import tensorflow as tf

class PolicyGradientActorCritic(object):

  def __init__(self, session,
                     state_dim,
                     num_actions,
                     init_exp=0.5,         # initial exploration prob
                     final_exp=0.0,        # final exploration prob
                     anneal_steps=10000,   # N steps for annealing exploration
                     discount_factor=0.99, # discount future rewards
                     reg_param=0.001,      # regularization constants
                     max_gradient=5,       # max gradient norms
                     summary_writer=None,
                     summary_every=100):

    # Environmental parameters
    self.state_dim = state_dim
    self.num_actions = num_actions

    # Tensorflow machinery
    self.session        = session
    self.optimizer      = None
    self.summary_writer = summary_writer
    self.summary_every  = summary_every

    # Model components
    self.actor_network = None
    self.critic_network = None

    # Training parameters
    self.discount_factor = discount_factor
    self.max_gradient    = max_gradient
    self.reg_param       = reg_param

    # Exploration parameters
    self.exploration  = init_exp
    self.init_exp     = init_exp
    self.final_exp    = final_exp
    self.anneal_steps = anneal_steps

    # Training iterator count
    self.train_iteration = 0

    # Reset memory buffer
    self.state_buffer  = []
    self.reward_buffer = []
    self.action_buffer = []

    # Record full reward history for normalization
    self.all_rewards = []
    self.max_reward_length = 1000000

    # Build model and set up summary writer
    self._build_model()  # create and initialize variables

  # ===============================================
  # Actor network -> computes policy(s)
  # ===============================================
  def _actor_network(self, states):
    W1 = tf.get_variable("W1", [self.state_dim, 20], initializer=tf.random_normal_initializer())
    b1 = tf.get_variable("b1", [20], initializer=tf.constant_initializer(0))
    h1 = tf.nn.tanh(tf.matmul(states, W1) + b1)
    W2 = tf.get_variable("W2", [20, self.num_actions], initializer=tf.random_normal_initializer(stddev=0.1))
    b2 = tf.get_variable("b2", [self.num_actions], initializer=tf.constant_initializer(0))
    p = tf.matmul(h1, W2) + b2
    return p

  # ===============================================
  # Critic network -> computes Q(s,a)
  # ===============================================
  def _critic_network(self, states):
    W1 = tf.get_variable("W1", [self.state_dim, 20], initializer=tf.random_normal_initializer())
    b1 = tf.get_variable("b1", [20], initializer=tf.constant_initializer(0))
    h1 = tf.nn.tanh(tf.matmul(states, W1) + b1)
    W2 = tf.get_variable("W2", [20, 1], initializer=tf.random_normal_initializer())
    b2 = tf.get_variable("b2", [1], initializer=tf.constant_initializer(0))
    v = tf.matmul(h1, W2) + b2
    return v

  # ===============================================
  # _build_model
  # ===============================================
  def _build_model(self):

    # Set optimizer and policy network
    self.optimizer = tf.train.RMSPropOptimizer(learning_rate=0.0001, decay=0.9)  # set optimizer
    self.actor_network = self._actor_network
    self.critic_network = self._critic_network

    # Input placeholder
    with tf.name_scope("model_inputs"):
      self.states = tf.placeholder(tf.float32, (None, self.state_dim), name="states")

    # Output for agent action sampling
    with tf.name_scope("predict_actions"):  # rollout action based on current policy
      with tf.variable_scope("actor_network"):
        self.policy_outputs = self.actor_network(self.states)
      with tf.variable_scope("critic_network"):
        self.value_outputs = self.critic_network(self.states)

      self.action_scores = tf.identity(self.policy_outputs, name="action_scores")  # predict action scores from policy network
      self.predicted_actions = tf.multinomial(self.action_scores, 1)  # sample action from multinomial distrib

    # Gather a list of trainable variables (regularization loss)
    actor_network_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="actor_network")
    critic_network_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="critic_network")

    # Compute policy gradients (via computing loss and gradients)
    with tf.name_scope("compute_pg_gradients"):

      # Placeholder for action and discounted rewards
      self.taken_actions = tf.placeholder(tf.int32, (None,), name="taken_actions")
      self.discounted_rewards = tf.placeholder(tf.float32, (None,), name="discounted_rewards")

      with tf.variable_scope("actor_network", reuse=True):
        self.logprobs = self.actor_network(self.states)
      with tf.variable_scope("critic_network", reuse=True):
        self.estimated_values = self.critic_network(self.states)

      # ===================================================
      # Actor gradients
      # ===================================================

      # Compute loss as policy gradient loss + regularization loss:
      #   L = L_pg + alpha_reg * L_reg
      # where:
      #   L_pg = mean(xentropy-loss)
      #   L_reg = sum(sum(||x||^2: every network variable))
      # Note:
      #   xentropy loss increases as predicted prob diverges from label -> perfect model has log loss of 0
      #   for example, it is bad if you predict 0.012 where the ground truth label is 1
      self.cross_entropy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logprobs,
                                                                               labels=self.taken_actions)
      self.pg_loss = tf.reduce_mean(self.cross_entropy_loss)
      self.actor_reg_loss = tf.reduce_sum([tf.reduce_sum(tf.square(x)) for x in actor_network_variables])
      self.actor_loss = self.pg_loss + self.reg_param * self.actor_reg_loss

      # Compute loss gradient vector for variables (W,b) in each layer
      #   ( D_theta(log(policy(s,a))) , var) list
      self.actor_gradients = self.optimizer.compute_gradients(self.actor_loss, actor_network_variables)
      # Compute advantages A(s) = R - V(s)
      self.advantages = tf.reduce_sum(self.discounted_rewards - self.estimated_values)
      # Compute policy gradient vector D_theta(J) for variables (W,b) in each layer
      #   ( D_theta(J) = D_theta(log(policy(s,a))) * R_discount(s,a) , var) list
      for i, (grad, var) in enumerate(self.actor_gradients):
        if grad is not None:
          self.actor_gradients[i] = (grad * self.advantages, var)

      # ===================================================
      # Critic gradients
      # ===================================================
      self.mean_square_loss = tf.reduce_mean(tf.square(self.discounted_rewards - self.estimated_values))
      self.critic_reg_loss = tf.reduce_sum([tf.reduce_sum(tf.square(x)) for x in critic_network_variables])
      self.critic_loss = self.mean_square_loss + self.reg_param * self.critic_reg_loss
      self.critic_gradients = self.optimizer.compute_gradients(self.critic_loss, critic_network_variables)

      # ===================================================
      # Full gradients
      # ===================================================
      self.gradients = self.actor_gradients + self.critic_gradients

      for i, (grad, var) in enumerate(self.gradients):
        if grad is not None:
          self.gradients[i] = (tf.clip_by_norm(grad, self.max_gradient), var)  # clip gradients by norm

      # ===================================================
      # Summary
      # ===================================================
      # Record distribution of variables and their gradients into histogram summary
      for grad, var in self.actor_gradients:
        tf.summary.histogram(var.name, var)
        if grad is not None:
          tf.summary.histogram(var.name + '/gradients', grad)

      # emit summaries
      tf.summary.histogram("estimated_values", self.estimated_values)
      tf.summary.scalar("actor_loss", self.actor_loss)
      tf.summary.scalar("critic_loss", self.critic_loss)
      tf.summary.scalar("reg_loss", self.actor_reg_loss + self.critic_reg_loss)

    # training update
    with tf.name_scope("train_actor_critic"):
      # apply gradients to update actor network
      self.train_step = self.optimizer.apply_gradients(self.gradients)

    # Merge summaries
    self.summarize = tf.summary.merge_all()
    self.no_op = tf.no_op()

    # Initialize all variables in the built model (will initialize when session is run)
    var_lists = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)  # collect variables
    self.session.run(tf.variables_initializer(var_lists))  # initialize variables
    self.session.run(tf.assert_variables_initialized())  # make sure all variables are initialized

    # Use summary writer if user provides it
    if self.summary_writer is not None:
      self.summary_writer.add_graph(self.session.graph)

  # ===============================================
  # get_action
  # ===============================================
  def get_action(self, states):

    # Softmax function for normalizing action scores for sampling (log probs)
    def softmax(y):
      maxy = np.amax(y)
      e = np.exp(y - maxy)
      return e / np.sum(e)

    # Epsilon-greedy exploration
    if random.random() < self.exploration:
      return random.randint(0, self.num_actions-1)  # random exploration
    else:
      action_scores = self.session.run(self.action_scores, {self.states: states})[0]  # compute action scores = log probs
      action_probs  = softmax(action_scores) - 1e-5  # perform softmax normalization on action scores -> action probs
      action = np.argmax(np.random.multinomial(1, action_probs))  # perform 1 sample experiment on action probs
      # actions = self.session.run(self.predicted_actions, {self.states: states})[0]  # sample action from policy
      # action = actions[0]
      return action

  # ===============================================
  # train
  # ===============================================
  def train(self):

    N = len(self.reward_buffer)  # length of reward buffer (length of one episode)
    r = 0 # use discounted reward to approximate Q value

    # Compute discounted future rewards
    discounted_rewards = np.zeros(N)
    for t in reversed(range(N)):
      # future discounted reward from now on
      r = self.reward_buffer[t] + self.discount_factor * r
      discounted_rewards[t] = r

    # Normalize discounted rewards to reduce gradient variance
    self.all_rewards += discounted_rewards.tolist()
    self.all_rewards = self.all_rewards[:self.max_reward_length]
    discounted_rewards -= np.mean(self.all_rewards)
    discounted_rewards /= np.std(self.all_rewards)

    # whether to calculate summaries
    calculate_summaries = self.summary_writer is not None and self.train_iteration % self.summary_every == 0

    # Update policy network using every element in replay memory buffer
    for t in range(N-1):

      # Prepare training input
      states  = self.state_buffer[t][np.newaxis, :]
      actions = np.array([self.action_buffer[t]])
      rewards = np.array([discounted_rewards[t]])

      # Perform one training step (compute policy gradient and optimizer uses it to make a step)
      _, summary_str = self.session.run([self.train_step,
                                         self.summarize if calculate_summaries else self.no_op],
                                        {self.states: states,
                                         self.taken_actions: actions,
                                         self.discounted_rewards:rewards})

      # Emit summaries
      if calculate_summaries:
        self.summary_writer.add_summary(summary_str, self.train_iteration)

    # After training, we anneal the exploration parameter to a lower value
    self._annealExploration()
    self.train_iteration += 1

    # Reset memory buffer
    self.reset_memory()

  def _annealExploration(self, stategy='linear'):
    ratio = max((self.anneal_steps - self.train_iteration)/float(self.anneal_steps), 0)
    self.exploration = (self.init_exp - self.final_exp) * ratio + self.final_exp

  # ===============================================
  # store
  # ===============================================
  def store(self, state, action, reward):
    self.action_buffer.append(action)
    self.reward_buffer.append(reward)
    self.state_buffer.append(state)

  # ===============================================
  # reset_model
  # ===============================================
  def reset_model(self):
    self.reset_memory()
    self.train_iteration = 0
    self.exploration = self.init_exp
    var_lists = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    self.session.run(tf.variables_initializer(var_lists))

  # ===============================================
  # reset_memory
  # ===============================================
  def reset_memory(self):
    self.state_buffer = []
    self.reward_buffer = []
    self.action_buffer = []
