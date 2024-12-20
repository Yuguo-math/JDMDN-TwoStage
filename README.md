# Official implementation and weights of the article : Joint demosaicking and denoising benefits from a two-stage training strategy
Joint demosaicking and denoising benefits from a two-stage training strategy

## Requirements
Pytorch

## Demo 
### Noise-free demosaicing 
	$ python demo_DM.py --mode 'normal' 
  $ python demo_DM.py --mode 'lightweight' 

### Fixed noise joint demosaicing-denoising (sigma=3, 5, 10, 15, 20, 40, and 60) 
	$ python demo_DMDN.py --mode 'normal' --sigma 20 
  $ python demo_DMDN.py --mode 'lightweight' --sigma 20 

### Flexible noise joint demosaicing-denoising (you can enter any value between 1-20) 
	$ python demo_DMDN_F.py --mode 'normal' --sigma 20 
  $ python demo_DMDN_F.py --mode 'lightweight' --sigma 20 
  
## Cite
Please cite the paper whenever JDMDN-TwoStage is used to produce published results or incorporated into other software:

	@article{guo2023joint, \
      title={Joint demosaicking and denoising benefits from a two-stage training strategy}, \
      author={Guo, Yu and Jin, Qiyu and Morel, Jean-Michel and Zeng, Tieyong and Facciolo, Gabriele}, \
      journal={Journal of Computational and Applied Mathematics}, \
      volume={434}, \
      pages={115330}, \
      year={2023}, \
      publisher={Elsevier} \
	}
