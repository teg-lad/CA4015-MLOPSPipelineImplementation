## Scoping

This file contains a scoping example of a potential real-world use case.

In practice, data pipelines are put in place by workflow management framework such as [Apache Airflow](https://airflow.apache.org/). We won't be implementing one of those, but once you have the scripts for processing data it is just a matter of scheduling the tasks.

### Potential Use Case

We will be looking at a [dataset](https://www.kaggle.com/datasets/nelgiriyewithana/emotions/data) containing Twitter/X messages annotated with the emotions of the speaker. These emotions are anger, fear, joy, love, sadness, and surprise.

We look at this data from the perspective of a machine learning engineer working at X who has been tasked with improving user engagement with ads. To do this, you have determined that understanding the users emotions on topics helps with ad targeting.

#### Problem Statement

The current problem is that the ads we are showing users don't seem to be giving us good click-through. So we need to figure out a better approach to ensure we are suggesting appropriate ads to users. If we can understand the emotion a user feels about a topic we can use that to help ad targeting and improve user response.

#### Data Needs

Creating a successful solutions will require user tweets/messages annotated with the displayed emotion. We likely don't need other data such as time or user data such as location, user id or gender. We could consult with subject-matter experts (SMEs) to determine if we have sufficient predictive signal. We may already have the needed raw data and just require labelling.

#### Proof of Concept

The initial proof of concept allows us to determine if it will be possible to create a model with the needed performance. We can create a [logistic regression model](logistic_regression.py) to quickly test if there is sufficient predictive signal in the [pre-processed](logistic_regression_preprocess.py) data. We can see that we have an accuracy of ~86%, so we can likely make a very accurate model.

#### Resources

Once we know the project is feasible, we need to set out some timeframes. It is likely we will take an iterative approach, with respect to data and modelling, so we can collect more data if we need to. We also need to consider compute resources, how big of a model do we need? Are there constraints? For us, we want the model to be small so that we can run it on every tweet, if the model is too big the overhead will be large.

#### Metrics

We may decide to use a hybrid of machine learning and business metrics to measure the success of the project. We may look at the micro average or the average across each class to determine if all classes have an acceptable performance from the machine learning side, then look at the click-through rate for the business metric.

#### Risks
Risks associated with this project include the potential biases in the emotion annotation process, which could affect the model's performance and fairness. Furthermore, deploying a biased model in ad targeting may lead to unintended consequences and negative user experiences, impacting brand reputation and user trust. It will be essential to mitigate these risks through careful data selection, preprocessing, and model evaluation to identify bias and allow us to remediate it.

---

Now that we have considered the scope of the project, we can move on to the Data stage.

