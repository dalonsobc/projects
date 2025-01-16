## Image Upscaling GAN Project

This project focuses on implementing and experimenting with a Generative Adversarial Network (GAN) for image upscaling, drawing inspiration from the work presented in. The primary goal is to replicate the basic functionality of an existing GAN for super-resolution, and to then explore modifications to both the generator and discriminator networks to assess their impact on the quality of upscaled images.

### Overview
The project implements a GAN for image upscaling, based on the architecture described in the linked paper. It begins with the implementation of the baseline model and then explores the effects of adding new convolutional layers to the generator and discriminator networks, as well as the possibility of outputting even higher-resolution images. The project uses a dataset of images that were then downsampled for input to the GAN, which is then tasked to upscale them back to the original resolution. The quality of the results of these experiments are then assessed visually.

### Key Components
Baseline GAN Implementation
* The initial step involves implementing a basic GAN for image upscaling, following the architecture described in the linked paper.
* This implementation serves as the baseline against which all subsequent modifications are compared.

Generator Modifications
* A convolutional layer was added to the generator to enhance feature refinement and detail extraction before upsampling.
* The impact of this addition on the quality of upscaled images was assessed by visual comparison.
* The modified generator produced outputs that were considered worse than the baseline, resulting in pattern images with a watermark of the original image.

Discriminator Modifications
* A convolutional layer was added to the discriminator to enable the capture of finer details in images, aiming to make it more difficult for the generator to produce realistic images.
* The impact of this addition on the quality of the upscaled images was assessed by visual comparison.
* The modified discriminator produced images similar to the baseline but with a sepia or vintage filter.
Higher Resolution Output
* An additional pixel-shuffle layer was added to the generator to upscale the images to a 512x512 resolution.
* Due to computational limitations, training for this higher resolution output was not completed, as the training time dramatically increased.
* Theoretically, the extra PixelShuffle layer is expected to upscale the images to higher resolutions while maintaining smoothness and reducing checkerboard artifacts.

Evaluation
* The evaluation of the different modifications was primarily based on visual inspection of the resulting images.
* There was a visual comparison to the baseline outputs, evaluating changes resulting from the generator and discriminator modifications.
* There was no sophisticated algorithm used for comparison purposes, the assessment was based on visual differences.

Results
* The baseline model was successfully implemented.
* The additional convolutional layer in the generator led to worse results than expected, producing pattern images with a watermark of the original.
* The additional convolutional layer in the discriminator led to a similar output to the baseline, but with a sepia or vintage filter.
* The attempt to upscale images to 512x512 resolution was aborted due to excessive computational time requirements, but theoretically should allow the generation of higher resolution images.

Conclusions
* The experiments highlight the sensitivity of GAN performance to architectural modifications.
* The modifications to the generator and discriminator did not result in an improvement in image quality.
* Careful hyperparameter selection and model design is crucial to achieve optimal GAN performance.
* Computational resources are a key limiting factor when training GANs for high-resolution images.