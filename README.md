What did I do in a nutshell?

In this project, we conducted a comprehensive analysis on the Iris dataset, a classic benchmark daset in machine learning. First, we explored the dataset, understanding its structure and variables. After dividing the dataset into training and testing sets to train and evaluate the models effectively, we employed the Recursive Partitioning and Regression Trees (rpart) algorithm to build a classification model for predicting iris species based on sepal and petal measurements. Following that, we implemented Naïve Bayes Classification, another popular algorithm in machine learning, using the same predictors. Finally, we compared the accuracy of both models and examined their confusion matrices to gain insights into their performance. Through this process, we aimed to demonstrate the effectiveness of these algorithms in classifying iris species and provide a comparative analysis of their predictive capabilities.

What data set did I use?

Provide the URL of the data set from Kaggle 
https://www.kaggle.com/datasets/arshid/iris-flower-dataset 

The Iris dataset is a classic and well-known dataset in the field of machine learning and statistics. It contains information about iris flowers, specifically measurements of the sepal length, sepal width, petal length, and petal width, as well as the corresponding species of each iris sample.

What did I find out?

Upon applying Recursive Partitioning and Regression Trees (rpart) and Naïve Bayes Classification to the Iris dataset, we discovered compelling insights into the predictive capabilities of these algorithms. The rpart model demonstrated a commendable accuracy of 95.56%, indicating its proficiency in classifying iris species based on sepal and petal measurements. Notably, the rpart model achieved a flawless classification rate when identifying Iris-setosa, showcasing its ability to accurately distinguish this species from others in the dataset. However, it encountered minor challenges when discriminating between Iris-versicolor and Iris-virginica, resulting in a total of two misclassifications. These misclassifications suggest potential areas for improvement in the model's decision boundaries to enhance its discriminatory power between closely related species.

In contrast, the Naïve Bayes Classification model exhibited even greater accuracy, achieving an impressive rate of 97.78%. This superior performance can be attributed to the model's robust probabilistic framework, which leverages the assumption of feature independence to make accurate predictions. Remarkably, the Naïve Bayes model achieved perfect classification for Iris-setosa, indicating its robustness in identifying this species accurately. However, similar to the rpart model, it encountered challenges in distinguishing between Iris-versicolor and Iris-virginica, resulting in a single misclassification. Despite this minor setback, the Naïve Bayes model demonstrated remarkable predictive capabilities, showcasing its effectiveness in classifying iris species based on the provided features.

Overall, both models exhibited robust predictive capabilities, with accuracy rates exceeding 95%. While the Naïve Bayes model demonstrated a marginal superiority in accuracy compared to the rpart model, both algorithms showcased similar misclassification patterns, particularly in discerning between Iris-versicolor and Iris-virginica. These findings highlight the challenges in accurately classifying closely related species based on limited features and highlight the importance of further research to refine and optimize classification algorithms for improved performance.

What does it mean?

The analysis of the Iris dataset using Recursive Partitioning and Regression Trees (rpart) and Naive Bayes Classification algorithms provided valuable insights into their performance in classifying iris species based on sepal and petal measurements.

Both models demonstrated high accuracy rates, highlighting their effectiveness for predictive modeling tasks. The rpart model achieved an impressive 95.56% accuracy in classifying iris species from the given features. However, it faced some difficulty distinguishing between Iris-versicolor and Iris-virginica, leading to a few misclassifications. This suggests that further refinement is needed to improve the decision boundaries for better separation of closely related species.

The Naive Bayes Classification model outperformed rpart with an accuracy of 97.78%. Its probabilistic framework, assuming feature independence, contributed to its superior performance. It perfectly classified Iris-setosa and performed well in differentiating Iris-versicolor and Iris-virginica, with only a single misclassification. Interestingly, both models exhibited similar misclassification patterns for closely related species, despite the Naive Bayes model's higher overall accuracy, indicating the inherent challenges posed by limited features.

The comparative analysis revealed a trade-off between model complexity and performance. While the Naive Bayes model achieved marginally higher accuracy, the rpart model offers interpretability through its decision tree structure, allowing users to understand the decision-making process. Therefore, the choice between these models depends on the specific application requirements, balancing predictive accuracy with interpretability.

In summary, the study demonstrated the effectiveness of rpart and Naive Bayes Classification algorithms for classifying iris species and provided insights into their performance characteristics. Future research could explore ensemble methods or feature engineering techniques to further improve classification accuracy and robustness, particularly for closely related species.

<img width="440" alt="image" src="https://github.com/user-attachments/assets/e85a21cf-2370-4a20-b54a-f2d93fa55c64">
<img width="514" alt="image" src="https://github.com/user-attachments/assets/087e9f09-6877-4fc4-b7fb-b30ec85595a7">
