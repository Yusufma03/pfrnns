# PF-RNNs

This is the PyTorch implementation of Particle Filter Recurrent Neural Networks (PF-RNNs).

Xiao Ma, Peter Karkus, David Hsu, Wee Sun Lee: [Particle Filter Recurrent Neural Networks. AAAI Conference on Artificial Intelligence (AAAI), 2020. ](https://arxiv.org/abs/1905.12885)

## Network structure

<img src="imgs/networks.jpg"/>

Above is the network structures for PF-LSTM and PF-GRU. In PF-RNNs, we maintain a set of latent particles and update them using particle filter algorithm. In our implementation, PF-LSTM and PF-GRU update particles in a parallel manner which benefit from the GPU acceleration.

## Install requirements
```
pip install -r requirements.txt
```

## Run the code
The training parameters are specified in configs/train.conf. To run the robot localization experiment, use
```
python main.py -c ./configs/train.conf
```

## Visualize particles
After training, you could visualize the particles by
```
python evaluate.py -c ./configs/eval.conf # save the latent particle tensors
python plot_particle.py --traj_num 0 --folder_num 0 # plot particles
```

## Acknowledgement
Thanks [Ta-Wei Yeh](https://github.com/TaWeiYeh) for inplementing the particle visualization code.

## Cite PF-RNNs
If you find this work useful, please consider citing us
```
@inproceedings{ma2020particle,
  author    = {Xiao Ma and
               P{\'{e}}ter Karkus and
               David Hsu and
               Wee Sun Lee},
  title     = {Particle Filter Recurrent Neural Networks},
  booktitle = {The Thirty-Fourth {AAAI} Conference on Artificial Intelligence, {AAAI}, 2020},
  pages     = {5101--5108}
}
```
