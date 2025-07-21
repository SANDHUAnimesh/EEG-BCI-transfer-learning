% Step 1: Use only feature 4 for training
selectedFeatures = 4;  % Select feature 4 only
bestFeatures = combinedFeatures(:, selectedFeatures);

% Step 2: Train LDA model using feature 4
ldaModel = fitcdiscr(bestFeatures, combinedLabels);

% Step 3: Train SVM model using feature 4
svmModel = fitcsvm(bestFeatures, combinedLabels, 'KernelFunction', 'rbf', 'BoxConstraint', 100, 'KernelScale', 1/0.1, 'Standardize', true);

% Step 4: Predict using both models on the test data with feature 4
testFeaturesSelected = testFeatures(:, selectedFeatures);  % Select feature 4 from the test data
ldaPredictedLabels = predict(ldaModel, testFeaturesSelected);
svmPredictedLabels = predict(svmModel, testFeaturesSelected);

% Step 5: Ensemble via Majority Voting
finalPredictedLabels = mode([ldaPredictedLabels, svmPredictedLabels], 2);  % Majority vote

% Step 6: Calculate test accuracy for the ensemble model using feature 4
ensembleTestAccuracy = mean(finalPredictedLabels == testLabels) * 100;

% Display the result
disp(['Test Accuracy with Ensemble (LDA + SVM) and Feature 4: ', num2str(ensembleTestAccuracy), '%']);
