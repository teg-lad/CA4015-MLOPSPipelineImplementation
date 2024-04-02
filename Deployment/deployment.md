## Deployment

This file contains some information on deploying a model to production, different deployment types and monitoring your model.

### Why do we deploy?

When we are developing a model we are working within development environments, using code editors and specialised
software to create a model. We need this software and packages in order to run the model. The end users will not know
how to do this as well as you do, they didn't create the model. Additionally, it makes it much better for their user
expierence if they can interact with your model in an easy way.

This is why deployment is important, it allows us to abstract the model running code and create a nice user interface
that can make use of an API (Application Programming Interface) to access your model. Creating user facing applications
may not be your responsibility, but it is good to know! Creating API's is more likely something you play a part in.

### What are some considerations?

Deploying a model comes with considerations. How much compute is needed to serve the model? If it is a big model there
will of course be more cost than a small model. How many requests/inference runs are you expecting? If you have a large
user base, you will likely need more compute to allow users to access your model. This is where latency comes in,
latency is how long it takes for the user to get the response from your model. Throughput is then the number of requests
you can process in a given unit of time.

You will find that there it is a balancing act, increasing compute raises costs, but lowers latency or allows for a
larger model. Having a smaller model means you need less compute, or you can have lower latency or better throughput.
There are lots of factors to consider!

You can choose to set up compute infrastructure yourself or have it in-house, this can be expensive to set up but gives
you more control. The alternative is doing the deployment on the cloud, this means that there isn't much to do in terms
of setup, and you can scale up or scale out as needed, giving you good flexibility.

### How do we deploy?

Deploying a model typically involves wrapping it in an API, this means that any program can call the model, regardless
of language, using just a HTTP request. Python has libraries such as Flask and FastAPI for this.

You may also decide to containerize the API. This basically means you create a Docker container that contains the API,
you can deploy a Docker container very easily on lots of different architectures without messing around with packages.
This means that you can quickly deploy a second instance if you get lots of traffic and your API is overloaded.

There are several strategies for deployment, these only really apply when you're deploying a new model to replace an old
one. These include:

+ Blue/Green Deployment - You create a duplicate the deployment architecture with the new model, switch the traffic from the Blue Deployment (old) to the Green Deployment (new).
  + This allows for quick rollback and no downtime
+ Canary Deployment  - You slow convert the existing endpoints to the new model. I.e. if you have 3 machines, change one to the new model, if all is well, change the 2nd and then the 3rd.
  + This means you don't need to expand the deployment architecture.

### Monitoring

Monitoring is what allows deployments to be agile, when we notice any model degradation we can prompty address it by
re-training the model.

The reality is that the world is constantly changing, meaning that our data and users are changing too. Unfortunately,
the data we train on is a snapshot of the world, and depending on the domain it can become stale fast. The model we have
trained perceives the world through the predictive signal that was in that data. So, changes in the data can cause our model to degrade as it doesn't understand how the world works **now**.

To help us quickly detect this, we set up monitoring to check the distribution of the data to detect data drift, which
is when there are changes  in the ranges and types of data we see. We can also label new data and evaluate the model and
rely on user feedback to determine if concept drift has occurred. Concept drift is when the data stays the same, but the
prediction is now of a different class.

Examples of data drift would be that people use new words in text classification models, or they change the way they
speak. Concept drift would be when world get re-defined or used in a new way. One example may be the word sick, I am so
sick we would think means unwell, but in recent times can be seen as 'cool', changing the meaning of the text, and
likely the prediction even though the data is the same.

When we detect that there is model degradation due to either data drift or concept drift, we can have the monitoring
workflow send us an alert to let us know we should investigate and take action.
