# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os

import numpy as np
from monai.transforms import (
    CastToTyped,
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    LoadImaged,
    NormalizeIntensityd,
    RandFlipd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandScaleIntensityd,
    EnsureTyped,
    CenterSpatialCropd,
    ConcatItemsd,
    DeleteItemsd,
    ANTsAffineRegistrationd,
    BrainExtractiond, Identityd,
)

from task_params import patch_size, deep_supr_num
from create_network import get_kernels_strides

from monai.transforms import SampleForegroundLocationsd, RandScaleIntensityFixedMeand, RandAdjustContrastd, \
    RandSimulateLowResolutiond, AppendDownsampledd, RandAffined


def get_task_transforms(mode,
                        task_id,
                        modality_keys,
                        pos_sample_num=None,
                        neg_sample_num=None,
                        use_nonzero=False,
                        registration_template_path=None,
                        preproc_out_dir=None,
                        do_brain_extraction=False,
                        use_prior=False,
                        ):

    label_keys = ["label"]
    all_keys = modality_keys + label_keys

    # exclude the prior from the intensity transforms
    mod_inty_keys = modality_keys[:-1] if use_prior else modality_keys

    load_image = LoadImaged(keys=all_keys, image_only=True)
    ensure_channel_first = EnsureChannelFirstd(keys=all_keys)
    crop_transform = CropForegroundd(keys=all_keys, source_key=mod_inty_keys[0], start_coord_key=None, end_coord_key=None)

    prep_load_tfm = Compose([load_image, ensure_channel_first, crop_transform], unpack_items=True)

    if mode == "prep":
        return prep_load_tfm

    elif mode in ["train", "validation"]:
        """ -1. add normalization to list of transforms based on the median of all crop size factors """
        norm_transform = NormalizeIntensityd(keys=mod_inty_keys, nonzero=use_nonzero)

        """ nnU-Net first calculates a larger patch-size, then samples the image across the image border (self.need_to_pad), so that the final
        smaller patch size (which is cropped from the center of the larger patch) will still cover the image borders.
        Here, instead, we directly achieve this by specifying the translate_range="cover" of the random affine. Internally,
        the translate_range is chosen such that after augmentation patches can have their center point as close as half the
        patch size from the image border"""

        """0. Oversample the foreground. This samples the foreground at a desired number of locations. Although the sampling is
        random, the transform doesn't inherit from RandomizableTransform, so that the result will automatically be cached for
        training. The sampled locations will be saved in the metadata and can be used by subsequent transforms."""

        # transform that creates a list of foreground locations (doesn't change the image/label data)
        sample_foreground_locations = SampleForegroundLocationsd(label_keys=label_keys, num_samples=10000)

        """## 1. Random affine transformation

        The following affine transformation is defined to output a (300, 300, 50) image patch.
        The patch location is randomly chosen in a range of (-40, 40), (-40, 40), (-2, 2) in x, y, and z axes respectively.
        The translation is relative to the image centre.
        The 3D rotation angle is randomly chosen from (-45, 45) degrees around the z axis, and 5 degrees around x and y axes.
        The random scaling factor is randomly chosen from (1.0 - 0.15, 1.0 + 0.15) along each axis.
        """

        ### Hyperparameters from experiment planning
        rotate_range = (30 / 360 * 2 * np.pi, 30 / 360 * 2 * np.pi, 30 / 360 * 2 * np.pi)
        scale_range = ((-0.3, 0.4), (-0.3, 0.4), (-0.3, 0.4))  # 1 is added to these values, so the scale factor will be (0.7, 1.4)

        rand_affine = RandAffined(
            keys=all_keys,
            mode=(3,)*len(modality_keys) + ("nearest", ),  # 3 means third order spline interpolation
            prob=1.0,
            spatial_size=patch_size[task_id],
            rotate_range=rotate_range,
            prob_rotate=0.2,
            translate_range=(0, 0, 0),
            foreground_oversampling_prob=pos_sample_num / neg_sample_num,  # None for random sampling according to translate_range, 1.0 for foreground sampling plus random translation according to translate_range, 0.0 for random within "valid" range
            label_key_for_foreground_oversampling="label",  # Determine which dictionary entry's metadata contains the sampled foreground locations
            prob_translate=1.0,
            scale_range=scale_range,
            prob_scale=0.2,
            padding_mode=("constant",)*len(modality_keys) + ("border", ),
        )

        """2. Gaussian Noise augmentation"""
        rand_gauss_noise = RandGaussianNoised(keys=mod_inty_keys, std=0.1, prob=0.1)

        """3. Gaussian Smoothing augmentation (might need adjustment, because image channels are smoothed independently in nnU-Net"""
        rand_gauss_smooth = RandGaussianSmoothd(keys=mod_inty_keys,
                                                sigma_x=(0.5, 1.0),
                                                sigma_y=(0.5, 1.0),
                                                sigma_z=(0.5, 1.0),
                                                prob=0.2 * 0.5, )  # 0.5 comes from the per_channel_probability

        """4. Intensity scaling transform"""
        scale_intensity = RandScaleIntensityd(keys=mod_inty_keys, factors=[-0.25, 0.25], prob=0.15)

        """5. ContrastAugmentationTransform"""
        shift_intensity = RandScaleIntensityFixedMeand(keys=mod_inty_keys, factors=[-0.25, 0.25], preserve_range=True,
                                                       prob=0.15)

        """6. Simulate Lowres transform"""
        sim_lowres = RandSimulateLowResolutiond(keys=mod_inty_keys, prob=0.25*0.5, zoom_range=(0.5, 1.0))

        """7. Adjust contrast transform with image inversion"""
        adjust_contrast_inverted = RandAdjustContrastd(keys=mod_inty_keys, prob=0.1 * 1.0, gamma=(0.7, 1.5),
                                                       invert_image=True, retain_stats=True)

        """8. Adjust contrast transform """
        adjust_contrast = RandAdjustContrastd(keys=mod_inty_keys, prob=0.3 * 1.0, gamma=(0.7, 1.5), invert_image=False,
                                              retain_stats=True)

        """9. Mirror Transform"""
        mirror_x = RandFlipd(all_keys, spatial_axis=[0], prob=0.5)
        mirror_y = RandFlipd(all_keys, spatial_axis=[1], prob=0.5)
        mirror_z = RandFlipd(all_keys, spatial_axis=[2], prob=0.5)

        """10. Downsampled labels"""
        _, strides = get_kernels_strides(task_id)

        supr_label_shapes = [patch_size[task_id]]
        for i in range(deep_supr_num[task_id]):
            last_shape = supr_label_shapes[-1]
            curr_strides = strides[
                i + 1]  # ignore first set of strides, since they apply to downsampling prior to the first level
            downsampled_shape = [int(np.round(last / curr)) for last, curr in zip(last_shape, curr_strides)]
            supr_label_shapes.append(downsampled_shape)

        if mode == "train":
            new_transform = Compose([
                norm_transform,  # -1
                sample_foreground_locations,  # 0
                rand_affine,  # 1
                rand_gauss_noise,  # 2
                rand_gauss_smooth,  # 3
                scale_intensity,  # 4
                shift_intensity,  # 5
                sim_lowres,  # 6
                adjust_contrast_inverted,  # 7
                adjust_contrast,  # 8
                mirror_x, mirror_y, mirror_z,  # 9
                AppendDownsampledd(label_keys, downsampled_shapes=supr_label_shapes),
                CastToTyped(keys=modality_keys, dtype=np.float32),
                EnsureTyped(keys=modality_keys),
                ConcatItemsd(keys=modality_keys, name="image", dim=0),
                DeleteItemsd(keys=modality_keys),
            ], unpack_items=True)

            return new_transform

        elif mode == "validation":
            transform = Compose([
                load_image,
                ensure_channel_first,
                crop_transform,
                norm_transform,  # -1
                CenterSpatialCropd(keys=all_keys, roi_size=patch_size[task_id]),  # to make sliding-window-inference much faster for validation (but restrict it to a central patch
                CastToTyped(keys=all_keys, dtype=(np.float32,)*len(modality_keys)+(np.uint8,)),
                EnsureTyped(keys=all_keys),
                ConcatItemsd(keys=modality_keys, name="image", dim=0),
                DeleteItemsd(keys=modality_keys),
            ], unpack_items=True)

            return transform

    elif mode == "test":
        print(f"{preproc_out_dir=}")
        print(f"{registration_template_path=}")
        affine_reg = ANTsAffineRegistrationd(keys=mod_inty_keys,  # register only the intensity image, the prior is already registered to the template
                                             moving_img_key=modality_keys[0],
                                             output_folder_path=os.path.join(preproc_out_dir, "registered"),
                                             template_path=registration_template_path)
        identity = Identityd(all_keys, allow_missing_keys=True)
        brain_extraction = BrainExtractiond(keys=mod_inty_keys, output_folder_path=os.path.join(preproc_out_dir, "brain_extracted"))
        load_image = LoadImaged(keys=modality_keys, image_only=True)
        ensure_channel_first = EnsureChannelFirstd(keys=modality_keys)
        crop_transform = CropForegroundd(keys=modality_keys, source_key=mod_inty_keys[0], start_coord_key=None, end_coord_key=None)
        norm_transform = NormalizeIntensityd(keys=mod_inty_keys, nonzero=use_nonzero)

        transform = Compose([
            affine_reg if registration_template_path else identity,
            brain_extraction if do_brain_extraction else identity,
            load_image,
            ensure_channel_first,
            crop_transform,
            norm_transform,
            # CenterSpatialCropd(keys=modality_keys, roi_size=patch_size[task_id]), # to make sliding-window-inference much faster for validation (but restrict it to a central patch
            CastToTyped(keys=modality_keys, dtype=(np.float32,) * len(modality_keys)),
            EnsureTyped(keys=modality_keys),
            ConcatItemsd(keys=modality_keys, name="image", dim=0),
            DeleteItemsd(keys=modality_keys),
        ], unpack_items=True)

        return transform


def determine_normalization_param_from_crop(prep_data_loader, key):
    '''
    Helper function to determine the "use_nonzero" parameter of the NormalizeIntensity transform. It loads the whole
    dataset via the provided dataloader, whose preprocessing transform includes the CropForeground transform. If the
    median volume reduction achieved by the cropping is more than 25%, use_nonzero will be returned as True, otherwise
    false.
    :param prep_data_loader: The dataloader, must be configured to go through the complete dataset once
    :param key: The key of the dictionaries that leads to the cropped MetaTensor
    :return: bool, that indicates if zeros should be included in normalization or not.
    '''

    all_crop_size_factors = []

    def get_crop_size_factor(img):
        # among applied transforms, find the idx of the crop transform
        for idx, tfm in enumerate(img.applied_operations):
            if tfm['class'] == 'CropForeground':
                break
        crop_size_factor = np.prod(img.shape) / np.prod(img.applied_operations[idx]['orig_size'])
        return crop_size_factor

    for batch in prep_data_loader:
        for sample in batch:
            all_crop_size_factors.append(get_crop_size_factor(sample[key]))

    print(f"{all_crop_size_factors=}")

    return True if np.median(all_crop_size_factors) < 0.75 else False