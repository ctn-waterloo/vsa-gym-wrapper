import numpy as np
from vsagym import spaces

def test_hex_sspspace():
    domain_dim = 2  #
    bounds = np.tile([-1, 1], (domain_dim, 1))
    ssp_type = 'hex'
    ssp_space = spaces.HexagonalSSPSpace(domain_dim,
                       n_scales=6, n_rotates=6,
                       domain_bounds=bounds, length_scale=0.1)

    x = np.array([0.1, -0.4])
    phi = ssp_space.encode(x)
    xhat = ssp_space.decode(phi, method='direct-optim')
    assert np.sqrt(np.sum((x - xhat) ** 2)) < 1e-5

def test_rand_sspspace():
    domain_dim = 2  #
    bounds = np.tile([-1, 1], (domain_dim, 1))
    ssp_type = 'hex'
    ssp_space = spaces.RandomSSPSpace(domain_dim,
                            ssp_dim=501, domain_bounds=bounds, length_scale=0.1)

    x = np.array([0.1, -0.4])
    phi = ssp_space.encode(x)
    xhat = ssp_space.decode(phi, method='direct-optim')
    assert np.sqrt(np.sum((x - xhat) ** 2)) < 1e-5

def test_sspbox():
    ssp_dim = 151
    box_space = spaces.SSPBox(-1, 1, 2, shape_out=ssp_dim, decoder_method='from-set', length_scale=None)
    box_space = spaces.SSPBox(-1, 1, 2, shape_out=ssp_dim, decoder_method='from-set', length_scale=0.1)
    x = np.array([0.1, -0.3])
    ssp = box_space.encode(x)
    box_space.sample();
    box_space.samples(2);

    ssp_space = spaces.HexagonalSSPSpace(2,
                                     n_scales=4, n_rotates=4,
                                     length_scale=0.1)

    box_space = spaces.SSPBox(-1, 1, 2,ssp_space=ssp_space)
    assert np.allclose(ssp, box_space.encode(x))

def test_sspdiscrete():
    ssp_dim = 151
    discrete_space = spaces.SSPDiscrete(3, shape_out=ssp_dim)
    assert discrete_space.decode(discrete_space.encode(1)) == 1
    discrete_space.sample();
    discrete_space.samples(2);

def test_sspsequence():
    ssp_dim = 151
    seq_space = spaces.SSPSequence(
        spaces.SSPBox(-1, 1, 2, shape_out=ssp_dim, decoder_method='from-set', length_scale=0.1),
        length=3)
    seq = np.array([[0.1, -0.3], [0, -0.1], [-0.2, 0.5]])
    seq_space.decode(seq_space.encode(seq.reshape(-1)));
    seq_space.sample();
    seq_space.samples(2);


def test_sspdict():
    ssp_dim = 151

    dict_space = spaces.SSPDict({
        "object": spaces.SSPDiscrete(6, shape_out=ssp_dim),
        "position": spaces.SSPBox(-10, 10, 2, shape_out=ssp_dim, length_scale=0.1,
                           decoder_method='from-set'),
        "velocity": spaces.SSPBox(-1, 1, 2, shape_out=ssp_dim, length_scale=0.1,
                           decoder_method='from-set')
    },
        static_spaces={"slots": spaces.SSPDiscrete(3, shape_out=ssp_dim)},
        seed=0)

    dict_space.samples(2);

    def map_to_dict(x):
        return {'object': int(x[0]), 'position': x[1:3], 'velocity': x[3:]}

    def map_from_dict(x_dict):
        x = np.zeros(5)
        x[0] = x_dict['object']
        x[1:3] = x_dict['position']
        x[3:] = x_dict['velocity']
        return x

    def encode(x, static_spaces):
        ssp = (x['object'] * static_spaces['slots'].encode(0) +
               x['position'] * static_spaces['slots'].encode(1) +
               x['velocity'] * static_spaces['slots'].encode(2))
        return ssp.v

    def decode(ssp, spaces, static_spaces):
        x = {}
        bind = static_spaces['slots'].ssp_space.bind
        inv_slots = static_spaces['slots'].ssp_space.inverse_vectors
        x['object'] = spaces['object'].decode(bind(inv_slots[0], ssp))
        x['position'] = spaces['position'].decode(bind(inv_slots[1], ssp))
        x['velocity'] = spaces['velocity'].decode(bind(inv_slots[2], ssp))
        return x

    dict_space.set_map_to_dict(map_to_dict)
    dict_space.set_map_from_dict(map_from_dict)
    dict_space.set_encode(encode)
    dict_space.set_decode(decode)

    vsa_embed = dict_space.encode([2, 8.1, 4.2, 0.3, -0.1])
    dict_space.decode(vsa_embed);


test_hex_sspspace()
test_rand_sspspace()
# test_sspbox()
# test_sspdiscrete()
# test_sspsequence()
# test_sspdict()