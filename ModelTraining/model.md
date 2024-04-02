# Model Training

This file contains an outline on model training within MLOPS.

### What does Model Training look like?

Model training is an iterative process in MLOps, you create an initial mode, learn from the mistakes you may have made
and take them into account when making your next model. Unfortunately, it is not very likely that you'll create the
ideal model on your first iteration, but you'll make gradual improvements that you can learn from!

### What considerations are there?

You should consider how the model will be used downstream, will there be compute limitations? This could be the case if
the model needs to run on an edge device or be embedded in an application. Alternatively, if you need to use the model
for many calls you may want it to be efficient and fast, so the smaller you can make it without sacrificing accuracy the
better.

The dimensionality of the data will also affect the model size, so using dimensionality reduction techniques can help to
preserve the data while decreasing the dimensionality. You should always keep in mind, mode dimensions likely means more
data until model convergence, so if a feature is not needed you're probably best off dropping it, or using feature
engineering to capture the information in a combined feature.

You should also consider transparency, which allows the internals of the model to be dissected and the model decisions
understood. This is incredibly important if your model is going to make decisions that have an effect on an
individuals' autonomy! Would you want a model to make a decision about your life without knowing why?

### What techniques are there for training?

#### Training Techniques

##### Neural Architecture Search (NAS)

Neural Architecture Search is a method to select a model that will perform well on the task. You can look at this on a
macro-level in terms of distinct model types, or on a micro-level with individual components of a model being added
together in different combinations.

##### Hyperparameter Optimization

Hyperparameters are a part of the model that cannot be tuned and has to be set before training starts, these are usually
things such as the learning rate, batch size, number of layers, and gradient clipping among others. Finding good values
for these parameters can be difficult, but techniques such as bayesian hyperparameter optimisation can help here. 

##### Transfer Learning

Transfer learning is an interesting technique, you take a base model that was trained by someone else with an open
license for use (be careful as some licenses are non-commercial only). This means that you can have a starting point to
continue on training, which is useful if pre-training takes lots of data and compute, such as for a large language model.

It is always a good idea to become familiar with the model before using it, review the training set if available and do
some sanity checks. Don't rush to release a model that you don't understand as it can act erratic.

##### Low fidelity estimates

Low fidelity estimates allow you to get a quick idea of how well a model may work on your dataset. This fits well with
hyperparameter tuning and neural architecture search. The premise is to lower the input in size, which for images would
involve down-scaling an image. The idea is that you get a rough estimation of how well the model can work, meaning you
can quickly discard poor performing models and begin full-training of the promising one/s.

#### Model Evaluation

Evaluating the model is useful to give you confidence in the ability of your model. Just be sure not to let data leak
from your training test into the test set, nice [paper](https://arxiv.org/abs/2309.08632) on this in the spirit of April Fools Day! :)

##### Slicing your data

Looking at top level results can obscure the performance of your model on less represented classes. You should attempt
identify different user types, taking these as slices to determine if your model has any issues with certain user types.
Under-performance or issues for one user class should be dealt with so all users have a fair experience!

##### Explainability

Deep learning models are ofter seen as black-box models, meaning that the inner workings are obscured. As a data
scientist you should try to lift the veil! Why does your model act the way it does? Does the reasoning it uses seem
sound? These can all help you determine if your model is paying attention to the right features and not ending up at the
correct prediction purely by chance.

There is a lot of work being done here, and there are model-agnostic methods (can be used with any model), and
model-specific. [Shapley values](https://www.datacamp.com/tutorial/introduction-to-shap-values-machine-learning-interpretability#)
are a great model-agnostic explanability technique which are based off of game theory. Model-specific methods usually
focus on the architecture of the model and use activations to compute the influence of input on outputs, or at
intermediate stages.

Explainability allows us to open up the model and make sure there isn't anything strange happening, and if there is, we
can (hopefully) find the cause and remediate it!

##### Model Benchmarks

Model benchmarks are a good way of quickly checking the performance of a model on a given task. You should take these with a grain of salt as there can be data leakage, but you may be able to compare the performance of different models well. Have a look [here](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard) for the HuggingFace LLM leaderboard that contains lots of benchmarks.
Benchmark model, sensitivity analysis and residual analysis.

##### Model Safety

Model safety is a big concern in recent years, especially with the AI Act coming into force towards the end of the year.
There are many different considerations and there are many toolkits out there for evaluating this.

Adversarial examples are a concern for model safety, there are some classic examples where gaussian noise is introduced
to images to change the predicted class. This cleverly uses the model weights to determine what changes in the input
cause a change in prediction. Feel free to check out this [paper](https://arxiv.org/abs/1412.6572)!

Bias and fairness are also important in your model. You should consider all the possible users and ensure that their
experiences are not different from one another. If a training set is representative of one user and not another, the
model may not perform as well for the under-represented user.


