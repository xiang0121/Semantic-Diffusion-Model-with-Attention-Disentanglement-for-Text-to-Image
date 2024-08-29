import itertools
from typing import Any, Callable, Dict, Optional, Union, List
import random
import spacy
import torch
from torch.utils.data import Dataset, DataLoader
import math
from diffusers import StableDiffusionPipeline, AutoencoderKL, UNet2DConditionModel
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput, StableDiffusionSafetyChecker
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import (
    EXAMPLE_DOC_STRING,
    rescale_noise_cfg
)
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_attend_and_excite import (
    AttentionStore,
    AttendExciteAttnProcessor
)
import numpy as np
from diffusers.schedulers import KarrasDiffusionSchedulers
from diffusers.utils import (
    logging,
    replace_example_docstring,
)
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer
from transformers import AutoProcessor, BlipForConditionalGeneration

logger = logging.get_logger(__name__)

# import utils
from utils.parsing import get_attention_map_index_to_wordpiece, split_indices, calculate_positive_pairs, calculate_negative_pairs, get_indices, start_token, end_token, align_wordpieces_indices, extract_attribution_indices, extract_attribution_indices_with_verbs, extract_attribution_indices_with_verb_root, extract_entities_only
from utils.load_caption_model import load_model
from utils.compute_loss import entropy_regularization, get_attention_map_index_to_wordpiece, split_indices, get_indices, start_token, end_token, align_wordpieces_indices, extract_attribution_indices, extract_attribution_indices_with_verbs, extract_attribution_indices_with_verb_root, extract_entities_only
from utils.mi_estimator import MI_Maximizer, MI_Minimizer

import time
from utils.arguments import parse_args

class CaptionModelWrapper(torch.nn.Module):
    def __init__(self, caption_model, weights, device, args, dtype):
        super().__init__()

        self.caption_model = caption_model
        self.model_name = caption_model
        
        self.device = device
        self.dtype = dtype
        self.caption_model_dict = {}
        load_device = device

        self.weights = {}
        self.args = args
        for model, weight in zip(caption_model, weights):
            self.weights[model] = weight


        load_model(self, caption_model, load_device, args)

    def forward(self, images, prompts, batch=None):
    
        caption_reward = self.blip_model.score(images, prompts, **batch)
     
        return caption_reward

class PairsSet(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __len__(self):
        
        return len(self.pairs)

    def __getitem__(self, idx):
        pair = self.pairs[idx]
        
        return pair[0].flatten(), pair[1].flatten()

class SemanticDiffusionPipeline(StableDiffusionPipeline):
    def __init__(self,
                 vae: AutoencoderKL,
                 text_encoder: CLIPTextModel,
                 tokenizer: CLIPTokenizer,
                 unet: UNet2DConditionModel,
                 scheduler: KarrasDiffusionSchedulers,
                 safety_checker: StableDiffusionSafetyChecker,
                 feature_extractor: CLIPImageProcessor,
                 requires_safety_checker: bool = False,
                 include_entities: bool = False,
                 ):
        super().__init__(vae, text_encoder, tokenizer, unet, scheduler, safety_checker, feature_extractor,
                         requires_safety_checker)
        
        self.parser = spacy.load("en_core_web_trf")
        self.subtrees_indices = None
        self.doc = None
        self.include_entities = include_entities
        self.caption_model = CaptionModelWrapper(caption_model="Blip", weights=[1.0], device="cuda", dtype=torch.float16, args=None)
        

    def _aggregate_and_get_attention_maps_per_token(self):
        attention_maps = self.attention_store.aggregate_attention(
            from_where=("up", "down", "mid"),
        )
        attention_maps_list = _get_attention_maps_list(
            attention_maps=attention_maps
        )
        return attention_maps_list

    @staticmethod
    def _update_latent(
            latents: torch.Tensor, loss: torch.Tensor, step_size: float, sr_scheduler
    ) -> torch.Tensor:
        """Update the latent according to the computed loss."""

        grad_cond = torch.autograd.grad(
            loss.requires_grad_(True), [latents], retain_graph=True, allow_unused=True
        )[0]    
        
        if grad_cond is None:
            print("Gradient is None. Skipping update.")
            return latents
        else:
            latents = latents - step_size *  (sr_scheduler * grad_cond)
        return latents

    def register_attention_control(self):
        attn_procs = {}
        cross_att_count = 0
        for name in self.unet.attn_processors.keys():
            if name.startswith("mid_block"):
                place_in_unet = "mid"
            elif name.startswith("up_blocks"):
                place_in_unet = "up"
            elif name.startswith("down_blocks"):
                place_in_unet = "down"
            else:
                continue

            cross_att_count += 1
            attn_procs[name] = AttendExciteAttnProcessor(
                attnstore=self.attention_store, place_in_unet=place_in_unet
            )

        self.unet.set_attn_processor(attn_procs)
        self.attention_store.num_att_layers = cross_att_count

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
            self,
            prompt: Union[str, List[str]] = None,
            height: Optional[int] = None,
            width: Optional[int] = None,
            num_inference_steps: int = 50,
            guidance_scale: float = 7.5,
            negative_prompt: Optional[Union[str, List[str]]] = None,
            num_images_per_prompt: Optional[int] = 1,
            eta: float = 0.0,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.FloatTensor] = None,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            negative_prompt_embeds: Optional[torch.FloatTensor] = None,
            output_type: Optional[str] = "pil",
            return_dict: bool = True,
            callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
            callback_steps: int = 1,
            cross_attention_kwargs: Optional[Dict[str, Any]] = None,
            guidance_rescale: float = 0.0,
            attn_res=(16,16),
            updating_rate: float = 20.0,
            parsed_prompt: str = None,
            num_intervention_steps: int = 25,
            semantic_refine = True,
            attention_disentanglement = True,
            args = None,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.
            height (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts to guide what to not include in image generation. If not defined, you need to
                pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
                to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            negative_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
                not provided, `negative_prompt_embeds` are generated from the `negative_prompt` input argument.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that calls every `callback_steps` steps during inference. The function is called with the
                following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function is called. If not specified, the callback is called at
                every step.
            cross_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the [`AttentionProcessor`] as defined in
                [`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            guidance_rescale (`float`, *optional*, defaults to 0.7):
                Guidance rescale factor from [Common Diffusion Noise Schedules and Sample Steps are
                Flawed](https://arxiv.org/pdf/2305.08891.pdf). Guidance rescale factor should fix overexposure when
                using zero terminal SNR.
            attn_res (`tuple`, *optional*, default computed from width and height):
                The 2D resolution of the semantic attention map.
            updating_rate (`float`, *optional*, default to 20.0):
                Controls the step size of each Semantic Diffusion update.
            num_intervention_steps ('int', *optional*, defaults to 20):
                The number of times we apply Semantic Diffusion.
            parsed_prompt (`str`, *optional*, default to None).


        Examples:

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] is returned,
                otherwise a `tuple` is returned where the first element is a list with the generated images and the
                second element is a list of `bool`s indicating whether the corresponding generated image contains
                "not-safe-for-work" (nsfw) content.
        """
        self.args = parse_args()
        prompt = self.args.prompt
        num_inference_steps = self.args.num_inference_steps
        num_intervention_steps = self.args.num_intervention_steps
        semantic_refine = self.args.semantic_refine
        attention_disentanglement = self.args.attention_disentanglement

        if parsed_prompt:
            self.doc = parsed_prompt
        else:
            self.doc = self.parser(prompt)
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor
        pos_pairs, neg_pairs = None, None
        # 1. Check inputs. Raise error if not correct
        self.check_inputs(
            prompt,
            height,
            width,
            callback_steps,
            negative_prompt,
            prompt_embeds,
            negative_prompt_embeds,
        )

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        text_encoder_lora_scale = (
            cross_attention_kwargs.get("scale", None) if cross_attention_kwargs is not None else None
        )
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
            negative_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            lora_scale=text_encoder_lora_scale,
        )
        # For classifier free guidance, we need to do two forward passes.
        # Here we concatenate the unconditional and text embeddings into a single batch
        # to avoid doing two forward passes
        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )
        # 6. Prepare extra step kwargs.
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        if attn_res is None:
            attn_res = int(np.ceil(width / 32)), int(np.ceil(height / 32))
        self.attn_res = attn_res
        self.attention_store = AttentionStore(self.attn_res)
        self.register_attention_control()

        text_embeddings = (
            prompt_embeds[batch_size * num_images_per_prompt:] if do_classifier_free_guidance else prompt_embeds
        )

        self.caption_model.to(device, dtype=torch.float16)
        
        # 7. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):

                # expand the latents if we are doing classifier free guidance
                latent_model_input = (
                    torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                )
                latent_model_input = self.scheduler.scale_model_input(
                    latent_model_input, t
                )

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=prompt_embeds,
                    cross_attention_kwargs=cross_attention_kwargs,
                    return_dict=False,
                )[0]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale * (
                            noise_pred_text - noise_pred_uncond
                    )

                if do_classifier_free_guidance and guidance_rescale > 0.0:
                    # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                    noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)

                
                if i < num_inference_steps:  
                    latents = self._semantic_diffusion_step(
                        latents,
                        noise_pred,
                        text_embeddings,
                        t,
                        i,
                        updating_rate,
                        cross_attention_kwargs,
                        prompt,
                        semantic_refine,
                        attention_disentanglement,
                        num_intervention_steps=num_intervention_steps,
                        
                    )

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]

                # call the callback, if provided
                if i == len(timesteps) - 1 or (
                        (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                ):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)
        
        # print(f"Time taken for {num_inference_steps} steps: {end_time - start_time}")
        if not output_type == "latent":
            image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
            image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)
            # has_nsfw_concept = None
        else:
            image = latents
            has_nsfw_concept = None

        if has_nsfw_concept is None:
            do_denormalize = [True] * image.shape[0]
        else:
            do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]

        image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(
            images=image, nsfw_content_detected=has_nsfw_concept
        ) #, pos_pairs, neg_pairs


    # ======================================================#
    #            Semantic Diffusion Step                    #
    # ===================================================== #
    def _semantic_diffusion_step(
            self,
            latents,
            noise_pred,
            text_embeddings,
            t,
            i,
            step_size,
            cross_attention_kwargs,
            prompt,
            semantic_refine,
            attention_disentanglement,
            num_intervention_steps,
    ):
        with torch.enable_grad():
            latents = latents.clone().detach().requires_grad_(True)
            updated_latents = []
            
            for latent, text_embedding in zip(latents, text_embeddings):
                # Forward pass of denoising with text conditioning
                latent = latent.unsqueeze(0)
                text_embedding = text_embedding.unsqueeze(0)

                self.unet(
                    latent,
                    t,
                    encoder_hidden_states=text_embedding,
                    cross_attention_kwargs=cross_attention_kwargs,
                    return_dict=False,
                )[0]
                self.unet.zero_grad()
                
                #========================== Semantic Refinement =============================#
                loss_s = 0
                if semantic_refine:
                    pred_x0 = self.pred_x0(latent, noise_pred, t, "cuda", text_embedding, "pt")
                    loss_s = self._compute_caption_loss(image=pred_x0, text=prompt)
                    gamma = self._exponential_scheduler(i, num_intervention_steps, 0.0, 0.01, 10)
                #============================================================================#
                
                loss_e = None
                loss_d = None
                loss_r = None
                #===================== Attention Disentanglement ============================#
                if attention_disentanglement:
                    # Get attention maps
                    loss_e = []
                    loss_d = []
                    loss_r = []

                    attention_maps = self._aggregate_and_get_attention_maps_per_token()
                    pos_pairs, neg_pairs, loss_r = self._compute_pairs(attention_maps=attention_maps, prompt=prompt)
                    
                    # compute mutual information
                    related_pairs = PairsSet(pos_pairs)
                    unrelated_pairs = PairsSet(neg_pairs)

                    mi_maximizer = MI_Maximizer()
                    mi_minimizer = MI_Minimizer()

                    for pos in related_pairs:
                        mi_maximizer.train(pos)
                    for neg in unrelated_pairs:
                        mi_minimizer.train(neg)

                    for pos in related_pairs:
                        loss_e.append(mi_maximizer.model(pos[0], pos[1]))
                    for neg in unrelated_pairs:
                        loss_d.append(mi_minimizer.model(neg[0], neg[1]))
                    
                loss_e = torch.stack(loss_e).mean() if loss_e else 0
                loss_d = torch.stack(loss_d).mean() if loss_d else 0
                loss_r = torch.stack(loss_d).mean() if loss_d else 0

                loss = ((self.args.related_disentangle_weight * loss_d - self.args.attribute_entangle_weight*loss_e) 
                                                    + self.args.entropy_regularization_weight * loss_r) + loss_s
                #================================================================================#
                
                # Perform gradient update
                if i < num_intervention_steps:
                    if loss != 0:
                        latent = self._update_latent(
                            latents=latent, loss=loss, step_size=step_size, sr_scheduler=gamma
                        )
                    logger.info(f"Iteration {i} | Loss: {loss:0.4f}")

            updated_latents.append(latent)

        latents = torch.cat(updated_latents, dim=0)

        return latents


    def _compute_caption_loss(self, image, text) -> torch.Tensor:
        reoslution = 512
        batch = {}
        batch["text"] = text

        # offset_range = reoslution // 224
        # random_offset_x = random.randint(0, offset_range)
        # random_offset_y = random.randint(0, offset_range)
        # size = reoslution - offset_range
        
        caption_rewards = self.caption_model(
            images=image,
            prompts=text, batch=batch)
        loss =  - caption_rewards
        # print(f"Loss: {loss}")
        return loss

    def _linear_scheduler(self, current_step, start_value, end_value, total_steps):
        # """Linear scheduler function."""
        # assert 0 <= current_step <= total_steps, "current_step should be within the range [0, total_steps]."
        # assert start_value <= end_value, "start_value should be less than or equal to end_value."

        slope = (end_value - start_value) / total_steps
        return start_value + slope * current_step

    def _exponential_scheduler(self, current_step, total_steps, min_value=0.0, max_value=1.0, alpha = 5):
        # """Exponentially increasing scheduler with configurable min and max values."""
        # a = Exponential growth factor
        if total_steps == 0:
            return min_value
        else:
            range_value = max_value - min_value
            max_exp_value = math.exp(alpha ) - 1
            exp_value = (math.exp(alpha  * (current_step / total_steps)) - 1) / max_exp_value
            return min_value + exp_value * range_value

    def _compute_pairs(
            self, attention_maps: List[torch.Tensor], prompt: Union[str, List[str]]
    ) -> torch.Tensor:
        attn_map_idx_to_wp = get_attention_map_index_to_wordpiece(self.tokenizer, prompt)
        pos_pairs, neg_pairs, entropy_loss = self._attribution_pairs(attention_maps, prompt, attn_map_idx_to_wp)

        # symmetrize the positive pairs
        sym_pos_pairs = [(pair[1], pair[0]) for pair in pos_pairs]
        pos_pairs = pos_pairs + sym_pos_pairs

        return pos_pairs, neg_pairs, entropy_loss


    def _attribution_pairs(
            self,
            attention_maps: List[torch.Tensor],
            prompt: Union[str, List[str]],
            attn_map_idx_to_wp,
    ) -> torch.Tensor:
        positive_pairs = []
        negative_pairs = []

        entropy_loss = 0

        if not self.subtrees_indices:
            self.subtrees_indices = self._extract_attribution_indices(prompt)
        subtrees_indices = self.subtrees_indices

        for subtree_indices in subtrees_indices:
            noun, modifier = split_indices(subtree_indices)
            all_subtree_pairs = list(itertools.product(noun, modifier))
            if noun and not modifier:
                if isinstance(noun, list) and len(noun) == 1:
                    processed_noun = noun[0]
                else:
                    processed_noun = noun
                _negative_pairs = calculate_negative_pairs(
                        attention_maps, modifier, processed_noun, subtree_indices, attn_map_idx_to_wp
                    )
                for pair in _negative_pairs:
                    negative_pairs.append(pair)
                entropy_loss += entropy_regularization(attention_maps, modifier, processed_noun)
            else:
                _positive_pairs, _negative_pairs, entropy_loss = self._calculate_map_pairs(
                    attention_maps,
                    all_subtree_pairs,
                    subtree_indices,
                    attn_map_idx_to_wp,
                )
                
                positive_pairs.extend(p for pair in _positive_pairs for p in pair)
                negative_pairs.extend(p for pair in _negative_pairs for p in pair)
                
        return positive_pairs, negative_pairs, entropy_loss


    def _calculate_map_pairs(
            self,
            attention_maps,
            all_subtree_pairs,
            subtree_indices,
            attn_map_idx_to_wp,
    ):
        positive_pairs = []
        negative_pairs = []
        entropy_loss= []
        for pair in all_subtree_pairs:
            noun, modifier = pair
            positive_pairs.append(
                calculate_positive_pairs(attention_maps, modifier, noun)
            )
            negative_pairs.append(
                calculate_negative_pairs(
                    attention_maps, modifier, noun, subtree_indices, attn_map_idx_to_wp
                )
            )
            entropy_loss.append(
                entropy_regularization(attention_maps, modifier, noun)
            )

        return positive_pairs, negative_pairs, entropy_loss
    #############################################

    def _align_indices(self, prompt, spacy_pairs):
        wordpieces2indices = get_indices(self.tokenizer, prompt)
        paired_indices = []
        collected_spacy_indices = (
            set()
        )  # helps track recurring nouns across different relations (i.e., cases where there is more than one instance of the same word)

        for pair in spacy_pairs:
            curr_collected_wp_indices = (
                []
            )  # helps track which nouns and amods were added to the current pair (this is useful in sentences with repeating amod on the same relation (e.g., "a red red red bear"))
            for member in pair:
                for idx, wp in wordpieces2indices.items():
                    if wp in [start_token, end_token]:
                        continue

                    wp = wp.replace("</w>", "")
                    if member.text.lower() == wp.lower():
                        if idx not in curr_collected_wp_indices and idx not in collected_spacy_indices:
                            curr_collected_wp_indices.append(idx)
                            break
                    # take care of wordpieces that are split up
                    elif member.text.lower().startswith(wp.lower()) and wp.lower() != member.text.lower():  # can maybe be while loop
                        wp_indices = align_wordpieces_indices(
                            wordpieces2indices, idx, member.text
                        )
                        # check if all wp_indices are not already in collected_spacy_indices
                        if wp_indices and (wp_indices not in curr_collected_wp_indices) and all(
                                [wp_idx not in collected_spacy_indices for wp_idx in wp_indices]):
                            curr_collected_wp_indices.append(wp_indices)
                            break

            for collected_idx in curr_collected_wp_indices:
                if isinstance(collected_idx, list):
                    for idx in collected_idx:
                        collected_spacy_indices.add(idx)
                else:
                    collected_spacy_indices.add(collected_idx)

            if curr_collected_wp_indices:
                paired_indices.append(curr_collected_wp_indices)
            else:
                print(f"No wordpieces were aligned for {pair} in _align_indices")
        # print(f"Paired indices collected:{paired_indices}")
        return paired_indices

    def _extract_attribution_indices(self, prompt):
        modifier_indices = []
        # extract standard attribution indices
        modifier_sets_1 = extract_attribution_indices(self.doc)
        modifier_indices_1 = self._align_indices(prompt, modifier_sets_1)
        if modifier_indices_1:
            modifier_indices.append(modifier_indices_1)

        # extract attribution indices with verbs in between
        modifier_sets_2 = extract_attribution_indices_with_verb_root(self.doc)
        modifier_indices_2 = self._align_indices(prompt, modifier_sets_2)
        if modifier_indices_2:
            modifier_indices.append(modifier_indices_2)

        modifier_sets_3 = extract_attribution_indices_with_verbs(self.doc)
        modifier_indices_3 = self._align_indices(prompt, modifier_sets_3)
        if modifier_indices_3:
            modifier_indices.append(modifier_indices_3)

        # entities only
        if self.include_entities:
            modifier_sets_4 = extract_entities_only(self.doc)
            modifier_indices_4 = self._align_indices(prompt, modifier_sets_4)
            modifier_indices.append(modifier_indices_4)

        # make sure there are no duplicates
        modifier_indices = unify_lists(modifier_indices)
        print(f"Final modifier indices collected:{modifier_indices}")
        return modifier_indices

    def pred_z0(self, sample, model_output, timestep):
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep].to(sample.device)

        beta_prod_t = 1 - alpha_prod_t
        if self.scheduler.config.prediction_type == "epsilon":
            pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
        elif self.scheduler.config.prediction_type == "sample":
            pred_original_sample = model_output
        elif self.scheduler.config.prediction_type == "v_prediction":
            pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output
            # predict V
            model_output = (alpha_prod_t**0.5) * model_output + (beta_prod_t**0.5) * sample
        else:
            raise ValueError(
                f"prediction_type given as {self.scheduler.config.prediction_type} must be one of `epsilon`, `sample`,"
                " or `v_prediction`"
            )

        return pred_original_sample
    
    def pred_x0(self, latents, noise_pred, t, device, prompt_embeds, output_type):

        
        pred_z0 = self.pred_z0(latents, noise_pred, t)
        
        pred_x0 = self.vae.decode(
            pred_z0 / self.vae.config.scaling_factor,
            return_dict=False,
        )[0]
        
        #pred_x0, ____ = self.run_safety_checker(pred_x0, device, prompt_embeds.dtype)
        do_denormalize = [True] * pred_x0.shape[0]
        
        pred_x0 = self.image_processor.postprocess(pred_x0, output_type=output_type, do_denormalize=do_denormalize)

        return pred_x0


def _get_attention_maps_list(
        attention_maps: torch.Tensor
) -> List[torch.Tensor]:
    attention_maps *= 100
    attention_maps_list = [
        attention_maps[:, :, i] for i in range(attention_maps.shape[2])
    ]

    return attention_maps_list


def unify_lists(list_of_lists):
    def flatten(lst):
        for elem in lst:
            if isinstance(elem, list):
                yield from flatten(elem)
            else:
                yield elem

    def have_common_element(lst1, lst2):
        flat_list1 = set(flatten(lst1))
        flat_list2 = set(flatten(lst2))
        return not flat_list1.isdisjoint(flat_list2)

    lst = []
    for l in list_of_lists:
        lst += l
    changed = True
    while changed:
        changed = False
        merged_list = []
        while lst:
            first = lst.pop(0)
            was_merged = False
            for index, other in enumerate(lst):
                if have_common_element(first, other):
                    # If we merge, we should flatten the other list but not first
                    new_merged = first + [item for item in other if item not in first]
                    lst[index] = new_merged
                    changed = True
                    was_merged = True
                    break
            if not was_merged:
                merged_list.append(first)
        lst = merged_list

    return lst

