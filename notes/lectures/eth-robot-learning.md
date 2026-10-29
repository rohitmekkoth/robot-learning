# ETH Robot Learning 2026 (Oier Mees)

Course: https://cvg.ethz.ch/lectures/Robot-Learning/
Videos: https://youtube.com/playlist?list=PLPU18BnWYUZJx3_d901-GD6BGpeWwE2vx
Homework: https://github.com/mees-robot-learning-course/ethz-course-2026

## L1 Introduction to Robot Learning

Source: Mees, O. (2026). *Lecture 1: Introduction to Robot Learning*. Robot Learning: From Fundamentals to Foundation Models, ETH Zürich. [Video](https://www.youtube.com/watch?v=X0k14u6pSxw). Notes are a paraphrased summary of the lecture; timestamps point to the video.

**Course shape** (6:00–21:00)
- Two halves: fundamentals (MDPs, imitation learning, online/offline RL) with individual homework in a MuJoCo SO-101 sim, then "scaling robot learning" (generative models, transformers, world models, generalist policies) with real-robot group projects on a fleet of SO-101 arms.
- Homework compute is laptop-scale; start early because policies take time to train.

**Why classical robotics hits a wall** (21:00–36:00)
- Lineage: Shakey (SRI, 1966) introduced the sense-plan-act loop; A* was developed for it. Industrial arms work the same way and succeed only in structured, fully known environments.
- Model-based humanoids (DARPA Robotics Challenge 2015, Atlas) rely on near-perfect state estimation, often via fiducials; small perception errors cause catastrophic failures.
- Moravec's paradox: abstract reasoning is easy for machines, sensorimotor skill is hard, partly because there is no internet-scale data for physical interaction.

**Definition** (36:00)
- Robot learning: algorithms that let a robot acquire skills or adapt to unstructured environments from data and experience instead of hand-coded programs. Short form: solving robotics with machine learning.
- Robotics supplies the "what" (perception, control); ML supplies the "how" (imitation, RL, representation learning).
- A robot needs embodiment and autonomy: a teleoperated surgical robot lacks autonomy, a chatbot lacks embodiment.

**Why now** (39:00–48:00)
- PR1 (Stanford, 2007) tidied rooms, but by teleoperation: the hardware existed; the missing piece was the policy.
- Three shifts since: commercial humanoids and capable hardware; GPU compute for scaling; data, though robotics data is far smaller than vision/NLP corpora.
- North star: generalist robot policies, one model mapping data to actions for any task, robot and environment.

**Robotics as multimodal sequence modeling** (48:00–54:00)
- Open X-Embodiment pooled data from 22 embodiments into one format; models such as Octo, RT-2 and π0 train on it.
- Key idea: a policy is a vision-language model whose outputs are action tokens instead of text, so VLM advances transfer to robotics.
- Why it is not "just train a big transformer": robot data is scarce and expensive (human teleoperation), heterogeneous across sensors and embodiments, inference must be real-time (model size matters), and real-world evaluation is slow and costly.

**Two algorithmic families** (54:00–57:00)
- Imitation learning: supervised learning from expert observation-action pairs; classical theory assumes i.i.d. data.
- Reinforcement learning: learn by trial and error to maximize reward. In robotics data is not i.i.d.: actions determine future states, creating a feedback loop.

**Takeaways for my roadmap**
- The i.i.d. vs. feedback-loop distinction is exactly why behavior cloning suffers distribution shift (CS 285 Lecture 2, DAgger in ETH HW3).
- "Policy = VLM with action outputs" is the framing behind the VLAs in the reading list (π0, SmolVLA, Octo).
- Real-world evaluation cost is the motivation for the flagship: predicting policy failure without rollouts.

**My questions**
-

## L2 Robot Control & MDPs

## L3 Imitation Learning

## L4 Reinforcement Learning I

## L5 Reinforcement Learning II

## L6 Generative Models

## L7 Sequence Modeling and Transformers

## L8 World Models

## L9 Generalist Robot Policies

## L10 Embodied Reasoning and Test-time Scaling

## L11 Frontier & Open Problems
