import numpy as np

class StateGenerator:

    def __init__(self, nrows=8, ncols=7, npieces=10):
        """
        Initialize a generator for sampling valid states from
        an npieces dimensional state space.
        """
        self.nrows = nrows
        self.ncols = ncols
        self.npieces = npieces
        self.rng = np.random.default_rng()

    def sample_state(self):
        """
        Samples a self.npieces length tuple.

        Output:
            Returns a state. A state is as 2-tuple (positions, dimensions), where
             -  Positions is represented as a list of position (c,r) tuples 
             -  Dimensions is a 2-tuple (self.nrows, self.ncols)

            For example, if the dimensions of the board are 2 rows, 3 columns, and the number of pieces
            is 4, then a valid return state would be ([(0, 0) , (1, 0), (2, 0), (1, 1)], (2,3))
        """
        ## Returns positions in decoded format. i.e. list of (c,r) i.e. (x,y)
        ## Without loss of generalization, we assume that positions[1:] are fixes; only
        ## positions[0] will be moved
        positions = self.rng.choice(self.nrows*self.ncols, size=self.npieces, replace=False)
        pos = list(self.decode(p) for p in positions)
        return pos, (self.nrows, self.ncols)

    def decode(self, position):
        r = position // self.ncols
        c = position - self.ncols * r
        return (c, r)

def get_cols(state):
    return state[1][1]

def get_pos(state):
    return state[0][0]

def get_pieces(state):
    return state[0][1:]

def get_rows(state):
    return state[1][0]

def is_oob(col, row, ncols, nrows):
    return col < 0 or col >= ncols or row < 0 or row >= nrows

def sample_observation(state):
    """
    Given a state, sample an observation from it. Specifically, the positions[1:] locations are
    all known, while positions[0] should have a noisy observation applied.

    Input:
        State: a 2-tuple of (positions, dimensions), the same as defined in StateGenerator.sample_state

    Returns:
        A tuple (position, distribution) where:
         - Position is a sampled position which is a 2-tuple (c, r), which represents the sampled observation
         - Distribution is a 2D numpy array representing the observation distribution

    NOTE: the array representing the distribution should have a shape of (nrows, ncols)
    """

    pos = get_pos(state)
    pieces = get_pieces(state)
    ncols = get_cols(state)
    nrows = get_rows(state)
    
    pos_n = tuple((pos[0], pos[1] - 1))
    pos_s = tuple((pos[0], pos[1] + 1))
    pos_e = tuple((pos[0] + 1, pos[1]))
    pos_w = tuple((pos[0] - 1, pos[1]))

    choices = list((pos, pos_n, pos_s, pos_e, pos_w))

    # create 1-D prob. dist.
    p = np.zeros(5)

    for i in range(len(p)):
        if i == 0:
            p[i] = 0.6
        else:
            # if an adjacent spot is OOB, we skip
            pc = choices[i]
            c = pc[0]
            r = pc[1]

            # if the adjacent spot is occupied, then
            # add 0.1 to pos location, otherwise
            # set to 0.1
            if is_oob(c, r, ncols, nrows):
                p[0] += 0.1
            elif pc in pieces:
                p[0] += 0.1
            else:
                p[i] = 0.1

    print('pieces', pieces)
    print('choices', choices, 'p', p)
    # sample a noisy observation based on the distribution
    sample = choices[np.random.choice(range(len(choices)), size=None, p=p)]
    
    # convert p to 2d array
    dist = np.zeros((nrows, ncols))
    for i in range(len(p)):
        pc = choices[i]
        c = pc[0]
        r = pc[1]

        # skip if adjacent piece is OOB
        if is_oob(c, r, ncols, nrows):
            continue

        dist[r, c] = p[i]

    return tuple((sample, dist))

def sample_transition(state, action):
    """
    Given a state and an action, 
    returns:
         a resulting state, and a probability distribution represented by a 2D numpy array
    If a transition is invalid, returns None for the state, and a zero probability distribution
    NOTE: the array representing the distribution should have a shape of (nrows, ncols)

    Inputs:
        State: a 2-tuple of (positions, dimensions), the same as defined in StateGenerator.sample_state
        Action: a 2-tuple (dc, dr) representing the difference in positions of position[0] as a result of
                executing this transition.

    Outputs:
        A 2-tuple (new_position, transition_probabilities), where
            - new_position is:
                A 2-tuple (new_column, new_row) if the action is valid.
                None if the action is invalid.
            - transition_probabilities is a 2D numpy array with shape (nrows, ncols) that accurately reflects
                the probability of ending up at a certain position on the board given the action. 
    """
    pos = get_pos(state)
    pieces = get_pieces(state)
    ncols = get_cols(state)
    nrows = get_rows(state)

    dc = action[0]
    dr = action[1]

    pos_c = pos[0]
    pos_r = pos[1]

    # get new position
    c_new = pos_c + dc
    r_new = pos_r + dr

    pos_new = tuple((c_new, r_new))

    if is_oob(c_new, r_new, ncols, nrows) or pos_new in pieces:
        pos_new = None
    
    # generate new prob dist.
    dist = np.zeros((nrows, ncols))
    if pos_new is not None:
        dist[r_new, c_new] = 1

    return tuple((pos_new, dist))
 
def initialize_belief(initial_state, style="uniform"):
    """
    Create an initial belief, based on the type of belief we want to start with

    Inputs:
        Initial_state: a 2-tuple of (positions, dimensions), the same as defined in StateGenerator.sample_state
        style: an element of the set {"uniform", "dirac"}

    Returns:
        an initial belief, represented by a 2D numpy array with shape (nrows, ncols)

    NOTE:
        The array representing the distribution should have a shape of (nrows, ncols).
        The occupied spaces (if any) should be zeroed out in the belief.
        We define two types of priors: a uniform prior (equal probability over all
        unoccupied spaces), and a dirac prior (which concentrates all the probability
        onto the actual position on the piece).
    
    """
    pos = get_pos(initial_state)
    pieces = get_pieces(initial_state)
    ncols = get_cols(initial_state)
    nrows = get_rows(initial_state)

    belief = np.zeros((nrows, ncols))
    if style == "uniform":
        # prob for an unoccupied space is 
        # 1 divided by the total number of unoccupied spaces
        total_spaces = nrows * ncols
        prob = 1 / (total_spaces - len(pieces))

        for r in range(nrows):
            for c in range(ncols):
                pc = tuple((c, r))  # pieces are ordered (col, row)
                if pc in pieces:
                    continue
                belief[r, c] = prob

    if style == "dirac":
        # prob should be 1 just at the
        # current position
        belief[pos[1], pos[0]] = 1 # pos is ordered (c, r), while the belief is ordered (r,c)

    return belief

def belief_update(prior, observation, reference_state):
    """
    Given a prior an observation, compute the posterior belief

    Inputs:
        prior: a 2D numpy array with shape (nrows, ncols)
        observation: a 2-tuple (col, row) representing the observation of a piece at a position
        reference_state: a 2-tuple of (positions, dimensions), the same as defined in StateGenerator.sample_state

    Returns:
        posterior: a 2D numpy array with shape (nrows, ncols)
    """
    ncols = get_cols(reference_state)
    nrows = get_rows(reference_state)

    posterior = prior
    
    # we need a new distribution based on the observation
    pieces = reference_state[0]
    np.insert(pieces, 0, observation)
    obs_state = tuple((pieces, reference_state[1]))
    obs = sample_observation(obs_state)
    obs_p = obs[1]

    # update all cells in the posterior
    sum = 0
    for r in range(nrows):
        for c in range(ncols):
            posterior[r, c] = obs_p[r, c] * posterior[r, c]
            sum += posterior[r, c]

    # normalize posterior
    for r in range(nrows):
        for c in range(ncols):
            posterior[r, c] = posterior[r, c] / sum

    return posterior

def belief_predict(prior, action, reference_state):
    """
    Given a prior, and an action, compute the posterior belief.

    Actions will be given in terms of dc, dr

   Inputs:
        prior: a 2D numpy array with shape (nrows, ncols)
        action: a 2-tuple (dc, dr) as defined for action in sample_transition
        reference_state: a 2-tuple of (positions, dimensions), the same as defined in StateGenerator.sample_state

    Returns:
        posterior: a 2D numpy array with shape (nrows, ncols)
    """
    ncols = get_cols(reference_state)
    nrows = get_rows(reference_state)
    
    posterior = np.zeros(nrows, ncols)
    transition = sample_transition(reference_state, action)

    for r in range(nrows):
        for c in range(ncols):
            # Bel[x] = p(x | x', u) * Bel[x']
            for r_prime in range(nrows):
                for c_prime in range(ncols):
                    posterior[r, c] += transition[r_prime, c_prime] * prior[r_prime, c_prime]

    return posterior

if __name__ == "__main__":
    gen = StateGenerator()
    initial_state = gen.sample_state()
    obs, dist = sample_observation(initial_state)
    print(initial_state)
    print(obs)
    print(dist)
    b = initialize_belief(initial_state, style="uniform")
    print(b)
    b = belief_update(b, obs, initial_state)
    print(b)
    b = belief_predict(b, (1,0), initial_state)
    print(b)
