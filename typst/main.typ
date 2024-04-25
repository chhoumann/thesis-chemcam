#import "@preview/charged-ieee:0.1.0": ieee
#import "@preview/glossarium:0.3.0": make-glossary, print-glossary, gls, glspl 
#import "@preview/whalogen:0.1.0": ce
#show: make-glossary

#set page(margin: 1.75in)
#set par(leading: 0.55em, first-line-indent: 1.8em, justify: true)
#set text(font: "New Computer Modern")
#show raw: set text(font: "New Computer Modern Mono")
#show par: set block(spacing: 0.55em)
#show heading: set block(above: 1.4em, below: 1em)

#show: ieee.with(
  title: [Title],
  abstract: [
    Abstract here
  ],
  authors: (
    (
      name: "Christian Bager Bach Houmann",
      department: [],
      organization: [Aalborg University],
      location: [Aalborg, Denmark],
      email: "chouma19@student.aau.dk"
    ),
    (
      name: "Patrick Frostholm Østergaard",
      department: [],
      organization: [Aalborg University],
      location: [Aalborg, Denmark],
      email: "pfas19@student.aau.dk"
    ),
    (
      name: "Ivik Lau Dalgas Hostrup",
      department: [],
      organization: [Aalborg University],
      location: [Aalborg, Denmark],
      email: "ihostr16@student.aau.dk"
    )
  ),
  index-terms: (),
  bibliography: bibliography("refs.bib"),
)

#print-glossary((
  (key: "msl", short: "MSL", long: "Mars Science Laboratory"),
  (key: "libs", short: "LIBS", long: "Laser-Induced Breakdown Spectroscopy"),
  (key: "chemcam", short: "ChemCam", long: "Chemistry and Camera"),
  (key: "mer", short: "MER", long: "Mars Exploration Rover"),
  (key: "pca", short: "PCA", long: "Principal Component Analysis"),
  (key: "pls", short: "PLS", long: "Partial Least Squares"),
  (key: "pls1-sm", short: "PLS1-SM", long: "Partial Least Squares 1 - sub-model"),
  (key: "ica", short: "ICA", long: "Independent Component Analysis"),
  (key: "moc", short: "MOC", long: "Multivariate Oxide Composition"),
  (key: "ann", short: "ANN", long: "Artificial Neural Network"),
  (key: "gbr", short: "GBR", long: "Gradient Boosting Regression"),
  (key: "rf", short: "RF", long: "Random Forest"),
  (key: "svm", short: "SVM", long: "Support Vector Machine"),
  (key: "svr", short: "SVR", long: "Support Vector Regression"),
  (key: "ksvr", short: "KSVR", long: "Kernelized Support Vector Regression"),
  (key: "lda", short: "LDA", long: "Linear Discriminant Analysis"),
  (key: "logreg", short: "LR", long: "Logistic Regression"),
  (key: "linreg", short: "LR", long: "Linear Regression"),
  (key: "lasso", short: "LASSO", long: "Least Absolute Selection and Shrinkage Operator"),
  (key: "ols", short: "OLS", long: "Ordinary Least Squares"),
  (key: "enet", short: "ENet", long: "Elastic Net"),
  (key: "omp", short: "OMP", long: "Orthogonal Matching Pursuit"),
  (key: "mulr", short: "MLR", long: "Multiple Linear Regression"),
  (key: "bpnn", short: "BPNN", long: "Back-Propagation Neural Network"),
  (key: "cnn", short: "CNN", long: "Convolutional Neural Network"),
  (key: "rmse", short: "RMSE", long: "Root Mean Squared Error"),
  (key: "rmsep", short: "RMSEP", long: "Root Mean Squared Error of Prediction"),
  (key: "rmsecv", short: "RMSE-CV", long: "Root Mean Squared Error of Cross-Validation"),
  (key: "mse", short: "MSE", long: "Mean Squared Error"),
  (key: "mae", short: "MAE", long: "Mean Absolute Error"),
  (key: "mdsbpnn", short: "MDS-BPNN", long: "Multidimensional Scaling-back Propagation Neural Network"),
  (key: "tas", short: "TAS", long: "Total Alkali-Silica"),
  (key: "pcr", short: "PCR", long: "Principal Component Regression"),
  (key: "fcnn", short: "FCNN", long: "Fully Connected Neural Network"),
  (key: "marscode", short: "MarSCoDe", long: "Mars Surface Composition Detector"),
  (key: "knn", short: "KNN", long: "K-Nearest Neighbors"),
  (key: "ulr", short: "ULR", long: "Univariate Linear Regression"),
  (key: "mvlr", short: "MLR", long: "Multivariate Linear Regression"),
  (key: "plsr", short: "PLSR", long: "Partial Least Squares Regression"),
  (key: "umap", short: "UMAP", long: "Uniform Manifold Approximation and Projection"),
  (key: "mad", short: "MAD", long: "Median Absolute Deviation"),
  (key: "k-elm", short: "K-ELM", long: "Kernel Extreme Learning Machine"),
  (key: "df", short: "DF", long: "Dominant Factor"),
  (key: "ccs", short: "CCS", long: "Clean, Calibrated Spectra"),
  (key: "uv", short: "UV", long: "Ultraviolet"),
  (key: "vio", short: "VIO", long: "Violet"),
  (key: "vnir", short: "VNIR", long: "Visible and Near-Infrared")
),
  disable-back-references: true
)

= Introduction
NASA has been studying the Martian environment for decades through a series of missions, including the Viking missions@marsnasagov_vikings, the #gls("mer") mission@marsnasagov_observer@marsnasagov_spirit_opportunity, and the #gls("msl") mission@marsnasagov_msl, each building on the knowledge gained from the previous ones.
Today, the rovers exploring Mars are equipped with sophisticated instruments for analyzing the chemical composition of Martian soil in search of past life and habitable environments.

Part of this research is facilitated through interpretation of spectral data gathered by #gls("libs") instruments, which fire a high-powered laser at soil samples to create a plasma.
The emitted light is captured by spectrometers and analyzed using machine learning models to assess the presence and concentration of certain major oxides, informing NASA's understanding of Mars' geology.

However, predicting major oxide compositions from #gls("libs") data presents significant computational challenges, including the high dimensionality and non-linearity of the data, compounded by multicollinearity and matrix effects@andersonImprovedAccuracyQuantitative2017. 
These effects can cause the intensity of emission lines from an element to vary independently of that element's concentration, introducing unknown variables that complicate the analysis. 
Furthermore, due to the high cost of data collection, datasets are often small, which further complicates the task of building accurate and robust models.

Various machine learning models have been used to predict the composition of major oxides in the sample, including #glspl("cnn") @yang_laser-induced_2022@yangConvolutionalNeuralNetwork2022, #gls("svr") @rezaei_dimensionality_reduction, and hybrid models like #gls("df")-#gls("k-elm") @song_DF-K-ELM that incorporate domain knowledge to enhance model interpretability and performance.
However, the high dimensionality and multicollinearity of the spectral data remains a significant challenge for these models.

Building upon the baseline established in #cite(<p9_paper>, form: "prose"), this thesis aims to explore approaches for tackling the challenges in predicting major oxide compositions from #gls("libs") data. We develop machine learning models that seek to enhance the accuracy and robustness of these predictions.
We define accuracy as the ability of a model to predict the composition of major oxides in Martian geological samples, while robustness refers to the stability of these predictions across different samples and oxides.

We investigate various techniques to handle the high dimensionality, non-linearity, and small dataset size inherent in this problem, and evaluate the performance of these models using appropriate metrics. 
Through extensive experiments on #gls("libs") data, we demonstrate the superior performance of our approach compared to existing methods in terms of both prediction accuracy and computational efficiency.

Our key contributions are as follows:
- We develop a novel machine learning pipeline that effectively handles the challenges of #gls("libs") data to accurately predict major oxide compositions in Martian soil samples.
- We conduct a comprehensive evaluation of various dimensionality reduction techniques and machine learning models to identify the optimal combination for this task. 
- We demonstrate the superior performance of our approach compared to existing methods through extensive experiments on #gls("libs") data.

// TODO: Add remaining sections
The remainder of this paper is organized as follows: 
Section~background provides background on the ongoing Mars exploration missions, the #gls("libs") technique, and the baseline #gls("moc") model. 
Section~@problem_definition formally defines the problem addressed in this work.
Section~@methodology describes our proposed methodology, including data preprocessing, dimensionality reduction, and machine learning models. 
Section~@experiments presents our experimental setup and results.
Finally, Section~@conclusion concludes the paper and discusses future work.

= Related Work <related_work>
In addressing the challenge of predicting major oxide compositions from #gls("libs") data, our investigation intersects with a broad spectrum of existing research that tackles similar computational hurdles, such as high dimensionality, multicollinearity, matrix effects, and the challenges of small datasets.
This section outlines how prior works relate to our problem, drawing from a range of techniques and methodologies that offer potential pathways for enhancing the accuracy and robustness of our predictions.

#cite(<andersonPostlandingMajorElement2022>, form: "prose") experimented with different machine learning models for quantifying major oxides on Mars using the SuperCam instrument on the Mars 2020 Perseverance rover.
They discuss preprocessing, normalization of #gls("libs") spectra, and the development of multivariate regression models to predict major element compositions.
For each oxide, they tested different models and selected the best performing one.
In some cases, they used a blend of models to improve the predictions.
The models they tested include: #gls("ols"), #gls("pls"), #gls("lasso"), Ridge, #gls("enet"), #gls("omp"), #gls("svr"), #gls("rf"), #gls("gbr"), and local #gls("enet") and blended submodels.
For $#ce("SiO2")$, they used a blend of #gls("gbr") and #gls("pls") models.
Interestingly, they found that PLS performed better at longer distances (4.25m), but GBR was better at 3m.
For $#ce("TiO2")$, they selected the #gls("rf") model for its superior performance at 4.25m and overall lower #gls("rmsep").
For $#ce("Al2O3")$, they used an average of predictions from four models (Local #gls("enet"), #gls("rf"), two variants of #gls("pls")) to obtain the lowest #gls("rmsep").
For $#ce("FeO_T")$, they initially selected #gls("rf") but later replaced it with #gls("gbr") due to its more realistic stoichiometry predictions for high-$#ce("Ca")$ pyroxenes and overall performance.
For $#ce("MgO")$, they selected #gls("gbr") for having the lowest #gls("rmsep") and avoiding negative predictions, despite slightly overpredicting $#ce("MgO")$ for high concentration samples.
For $#ce("CaO")$, they used a blend of #gls("rf") and #gls("pls") to address the bimodal distribution of $#ce("CaO")$ predictions by the #gls("rf") model alone.
For $#ce("Na2O")$, they used a blend of #gls("gbr") and #gls("lasso") models to utilize #gls("gbr")'s accuracy at low concentrations and #gls("lasso")'s superior predictions at higher concentrations.
For $#ce("K2O")$, they selected #gls("lasso") for its better performance on high $#ce("K2O")$ samples, despite the averaged model of five algorithms showing slight improvements at lower concentrations.
The findings of this paper are significant to us because they provide a benchmark for the performance of different machine learning models on #gls("libs") spectra.
We can use this information to guide our model selection and to compare our results with theirs.
Additionally, we might want to try out different models from the ones they tested to see if we can improve the predictions further, or perhaps find a model that is more suitable for our specific use case.
Also, SuperCam being the successor to #gls("chemcam") means that the findings of this paper are directly relevant to our work.

#cite(<song_DF-K-ELM>, form: "prose") present a novel approach to enhance the performance and interpretability of machine learning models in the context of #gls("libs") quantification.
The authors use "knowledge-based spectral lines, related to analyte compositions, to construct a linear physical principle based model and adopts #gls("k-elm") to account for the residuals of the linear model."
The method is based on #gls("df") and #gls("k-elm") and is called #gls("df")-\gls{k-elm}.
This method stands out by offering an intuitive explanation of how knowledge-based spectral lines impact prediction results, thereby enhancing model interpretability without compromising model complexity.
#gls("df")-#gls("k-elm") was tested across 10 regression tasks based on 3 #gls("libs") datasets, comparing its performance against six baseline methods using #gls("rmsep") as the evaluation metric.
They have 3 coal datasets, and they do regression tasks involving carbon, ash, volatile matter, and heat value analysis.
It achieved the best performance in 4 tasks and the second-best in 2 tasks, demonstrating its efficacy.
Incorporation of domain knowledge not only improved the accuracy of the models but also enhanced their generalizability across different tasks.
The method's design allows for a more interpretable machine learning model that adheres closer to the physical principles underlying #gls("libs") quantification.
The integration of domain knowledge into machine learning models addresses two critical challenges: improving the interpretability of complex models and enhancing their performance by leveraging specific domain insights.
The approach demonstrates a practical application of kernel extreme learning machines combined with domain-specific insights.
This is particularly valuable in fields like spectroscopy, where understanding the relationship between the spectral data and the analyte concentration is vital.
The #gls("df")-#gls("k-elm") method showcases how hybrid models can outperform traditional machine learning approaches.
The approach demonstrates a practical application of kernel extreme learning machines combined with domain-specific insights.
This is very relevant to our work considering interpretability is a key requirement for NASA and something they considered when choosing the #gls("pls") model for the #gls("chemcam") instrument.

#cite(<rezaei_dimensionality_reduction>, form: "prose") explore a variety of statistical and machine learning methods, including #gls("mulr"), #gls("svr"), #gls("ksvr"), and #gls("ann"), alongside their integrations with #gls("pca") to reduce dimensionality and improve model performance.
They use #gls("mse") and #gls("mae") as evaluation metrics to compare the performance of the models.
This paper clearly demonstrates the effectiveness of dimensionality reduction techniques in improving the performance of machine learning models because it compares the performance of many models with and without #gls("pca"): #gls("ann"), #gls("mulr"), #gls("svr"), #gls("ksvr"), #gls("pca")-\gls{ann}, #gls("pca")-#gls("mulr"), #gls("pca")-#gls("svr"), and #gls("pca")-#gls("ksvr").
For all elements, a variant of #gls("svr") performs the best.
For $#ce("Si")$, #gls("svr") performs the best.
For $#ce("Zn")$, #gls("pca")-#gls("svr") performs the best.
For the rest of the elements, #gls("pca")-#gls("ksvr") performs the best.
The superiority of #gls("ksvr") is attributed to the its ability to handle non-linear relationships in the data effectively, especially when combined with #gls("pca")'s capability to compress and simplify the input data by focusing on the most relevant variations.

#cite(<yang_laser-induced_2022>, form: "prose") present a study on the application of a deep #gls("cnn") for classifying geochemical samples using #gls("libs"), with a particular focus on planetary exploration missions such as China's Tianwen-1 Mars mission.
The authors demonstrate the effectiveness of a deep CNN in classifying geochemical standard samples using #gls("libs") spectra collected at varying distances.
This addresses the challenge of spectral differences induced by distance, showcasing that #gls("cnn") can learn to classify samples without the need for traditional spectral preprocessing or distance correction.
Using a dataset of over 18,000 #gls("libs") spectra from 39 geochemical standard samples, the study compares the #gls("cnn") model's performance against four other machine learning models: #gls("bpnn") #gls("svm"), #gls("lda"), and #gls("logreg").
The #gls("cnn") model exhibits superior classification accuracy, emphasizing its potential for geochemical sample identification/classification in planetary exploration.
The paper includes a detailed comparative analysis, proving the #gls("cnn") model's superior performance.
With classification accuracies on the validation set for all models exceeding 95%, the #gls("cnn") model demonstrated the highest overall accuracy.
This was particularly evident as the training set size increased, indicating the model's robustness to varying distances without requiring distance correction.
Statistical analysis further confirmed the #gls("cnn") model's superiority, with higher average Ncorr values compared to other models.
The #gls("cnn") model's ability to accurately classify geochemical samples without preprocessing for distance correction is quite impressive.
This is particularly relevant to our work because we are also working with #gls("libs") spectra collected at varying distances.
The comparative analysis underscores the #gls("cnn") model as a best-fit approach for #gls("libs") data analysis, potentially setting a new standard for future research and applications in the field.

#cite(<jeonEffectsFeatureEngineering2024>, form: "prose") investigated the effects of feature engineering on the robustness of #gls("libs") for steel classification.
They developed a remote #gls("libs") system to classify six steel types, using various feature-engineering and machine learning algorithms, including #gls("svm") and #gls("fcnn"), to handle different laser energies in test datasets.
They found that using intensity ratios as input data resulted in more robust classification models.
It was better than #gls("pca") and #gls("rf")-based wavelength selection.
The study highlights the importance of selecting appropriate feature engineering methods to improve model robustness, especially under varying measurement conditions.
This is relevant to our project as it demonstrates how feature engineering can enhance the performance and robustness of models for classifying materials based on #gls("libs") data, addressing challenges similar to those we face in predicting major oxide compositions.

The study by #cite(<fontanaLaserInducedBreakdown2023>, form: "prose").
explores using #gls("libs") for whole-rock geochemical analysis, specifically for major elements like $#ce("Al")$, $#ce("Ca")$, $#ce("Fe")$, $#ce("K")$, $#ce("Mg")$, $#ce("Na")$, $#ce("Si")$, and $#ce("Ti")$.
They averaged #gls("libs") spot analyses over 1-mm spaced transects on drill core intervals, demonstrating strong correlations with lab-based geochemistry for elements like $#ce("Si")$, $#ce("Al")$, and $#ce("Na")$.
Different predictive models were used for each element, such as #gls("pls"), #gls("enet"), #gls("lasso"), and #gls("pcr"), showing varied #gls("rmsecv") values indicating the precision of these models.
This method's relevance to our work lies in its potential for rapid, in-situ geochemical analysis, offering a way to overcome challenges related to high dimensionality and non-linearity in #gls("libs") data.

The paper by #cite(<sunMachineLearningTransfer2021>, form: "prose") introduces transfer learning to #gls("libs") spectral data analysis for rock classification on Mars, significantly improving model performance.
Previously, models trained on laboratory standards (pellets) struggled with physical matrix effects when applied to natural rock spectra.
Transfer learning, leveraging knowledge from one domain to address related problems in another, was applied to overcome this challenge.
The method showed remarkable improvement in #gls("tas") classification accuracy for both polished and raw rock samples, with rates increasing from 25% and 33.3% to 83.3% respectively using machine learning models to 83.3% with the transfer learning model.
This demonstrates the effectiveness of transfer learning in addressing the physical matrix effect and enhancing model robustness for rock classification on Mars.

#cite(<wangDeterminationElementalComposition2023>, form: "prose") discusses an advanced methodology for analyzing stream sediments using remote #gls("libs") combined with a #gls("mdsbpnn") algorithm.
This approach yielded highly accurate quantitative analyses of both major and trace elements, with determination coefficients (R2) for major elements exceeding 0.9996 and for trace elements greater than 0.9837, and #gls("rmse") less than 0.73 (major elements).
The study emphasizes the potential of remote #gls("libs") technology, especially for identifying biominerals in geological samples, highlighting its significance for studying ancient planetary environments.

// We had this one last semester IIRC. Might be good to have again?
#cite(<leporeQuantitativePredictionAccuracies2022a>, form: "prose") provides an in-depth analysis of the effectiveness of using #gls("libs") for geochemical analysis, focusing on the optimization of calibration datasets through the creation of submodels.
It outlines the methodology for collecting and processing #gls("libs") spectra, the development of multivariate models for predicting geochemical compositions, and compares the predictive accuracies of different submodel strategies.
The study finds that while submodels can improve prediction accuracies under certain conditions, the overall effectiveness is contingent upon having a large and diverse calibration dataset.
The research suggests that the optimal use of #gls("libs") for geochemical analysis requires a balance between the specificity of submodels and the breadth of the calibration dataset to ensure accurate and reliable predictions.

#cite(<kepesImprovingLaserinducedBreakdown2022>, form: "prose") discusses enhancing #gls("libs") model accuracy using transfer learning between ChemCam and SuperCam instruments.
It proposes a method where ChemCam data transforms to approximate SuperCam spectra, improving #gls("cnn") regression models' performance.
Key methods include data augmentation and fine-tuning of #gls("cnn")s with pre-processed and normalized spectra.
This approach outperforms some existing models for specific oxides, demonstrating transfer learning's potential in #gls("libs") analyses for more accurate quantitative models.

#cite(<ferreiraComprehensiveComparisonLinear2022>, form: "prose") presents an extensive comparison of various algorithms for quantifying lithium in geological samples, with a focus on both linear and non-linear methods.
The study tested algorithms on spectra acquired from a commercial handheld device and a laboratory prototype, highlighting the challenges in quantifying lithium due to effects like saturation and matrix interference.
The results showed that non-linear methodologies, such as #gls("knn") regression, #gls("svr"), and #gls("ann") regression, generally outperformed linear methods by effectively managing saturation and matrix effects, which are common in geological samples.
This research provides valuable insights for future applications in geological sample analysis and could potentially be generalized for other elements in similar contexts.
The paper's findings are particularly relevant to our project as it demonstrates the effectiveness of non-linear machine learning techniques in handling complex, non-linear relationships in high-dimensional #gls("libs") data, aligning with our research objectives of improving major oxide composition predictions from #gls("libs") data.

The study by #cite(<liuComparisonQuantitativeAnalysis2022>, form: "prose") explores the use of #gls("marscode") #gls("libs") for quantitative analysis of olivine in a simulated Martian atmosphere, focusing on multivariate analysis methods to address challenges posed by #gls("libs") data, such as high dimensionality and multicollinearity.
The methods evaluated include #gls("ulr"), #gls("mvlr"), #gls("pcr"), #gls("plsr"), ridge regression, #gls("lasso"), #gls("enet"), and #gls("bpnn").
The findings demonstrate the effectiveness of dimension reduction techniques, especially #gls("plsr"), and nonlinear analysis for improving quantitative analysis accuracy of olivine using #gls("libs") data.
This approach is particularly relevant to our work due to the focus on advanced statistical methods and machine learning algorithms for handling complex, high-dimensional #gls("libs") data, aligning with our objectives of improving accuracy and robustness in predicting major oxide compositions.

#cite(<yangConvolutionalNeuralNetwork2022>, form: "prose") is a study where a #gls("cnn") model is designed to identify twelve types of rocks using #gls("libs") data from the #gls("marscode") for the Tianwen-1 Mars exploration mission.
The classification performance of the #gls("cnn") is compared with #gls("logreg"), #gls("svm"), and #gls("lda").
The #gls("cnn") model achieved the highest classification accuracy, demonstrating its efficiency in rock identification with #gls("libs") spectra collected in a simulated Martian environment.
This indicates that #gls("cnn")-supported #gls("libs") classification is a promising analytical technique for planetary exploration missions.

#cite(<silvaRobustCalibrationModels2022>, form: "prose") introduce clustered regression calibration algorithms for #gls("libs") to address quantification challenges in complex sample matrices or wide concentration ranges, focusing on lithium quantification in geology.
They employ unsupervised clustering to group similar samples before applying a tailored linear calibration model to each cluster.
This approach, tested on lithium in exploration drills, outperforms standard linear models, especially in lower concentration regions, and demonstrates good generalizability to unseen data from different rock veins.
The study highlights the potential of clustered regression methods in improving #gls("libs") quantification accuracy and robustness, particularly valuable in mining environments.
Uses Ridge regression, #gls("pls"), dimensionality reduction with #gls("umap") and then k-means clustering, followed by a local model for each cluster, univariate calibration curves.

#cite(<woldPrincipalComponentAnalysis1987>, form: "prose") provides an in-depth tutorial on #gls("pca"), a fundamental method in multivariate data analysis used for dimensionality reduction, outlier detection, and uncovering the underlying structure in data sets.
The paper covers the history and development of #gls("pca"), its mathematical foundations, applications across various fields, and detailed instructions for data pre-treatment and #gls("pca") implementation.
It emphasizes #gls("pca")'s utility in simplifying complex data sets, enhancing interpretability, and supporting data analysis through the extraction of principal components that capture the most variance in the data.
This work is particularly relevant to our project as it underscores the importance of #gls("pca") in handling high-dimensional data, which aligns with our objectives of improving the prediction accuracy and interpretability of major oxide compositions from #gls("libs") data by effectively managing multicollinearity and high dimensionality challenges.

#cite(<bankAutoencoders2021>, form: "prose") details various autoencoder architectures and their applications in machine learning, emphasizing their role in dimensionality reduction, feature learning, and generative models.
It explores regularized, denoising, and variational autoencoders, highlighting their advantages in compressing data into lower-dimensional spaces while retaining essential information for reconstruction.
Autoencoders could significantly enhance our ability to handle the high-dimensional nature of #gls("libs") data.
By compressing spectral data into more manageable representations without losing critical information, we can improve model performance, especially in predicting major oxide compositions, by focusing on the most relevant features extracted from the compressed data representation.
This aligns with our objectives of efficient dimensionality reduction and robust predictive modeling.

// Multi-task learning
In their work #cite(<caruana_no_1997>, form: "prose"), presents a method called _Multitask learning_, which is a method of learning machine models on several related datasets.
The motivation for this approach stems from the assumption that utilizing multiple, albeit related, datasets can enhance a model's ability to discern patterns and shapes within the data.
#cite(<caruana_no_1997>, form: "prose") suggests that leveraging shared representations for model training can enable the model to identify underlying attributes in other datasets, even when this new data is small.
This is relevant for our work as one of the major challenges in analyzing LIBS calibration data for Mars is the scarcity of available data.
This scarcity makes it difficult to construct robust models capable of comprehensively understanding the underlying patterns and physical principles within the data.
Utilizing related LIBS data could help the models first learn the general outline, shape and patterns in the LIBS data, making it easier for it to grasp the deeper patterns in the Mars related data.


= Background <background>

= Problem Definition <problem_definition>

= Methodology <methodology>

= Experiments <experiments>

= Conclusion <conclusion>

