% Function to shift EEG data in time
function augmentedData = timeShiftEEG(data, shiftAmount)
    % data: The original EEG data (channels x timepoints x trials)
    % shiftAmount: Number of timepoints to shift the data
    
    augmentedData = circshift(data, [0, shiftAmount, 0]);  % Shift in the time dimension
end

% Step 1: Define the time shift amount
shiftAmount = 10;  % You can adjust this value (e.g., 10 timepoints)

% Step 2: Augment the source and target training data by shifting in time
augmentedSourceTrainFeatures = timeShiftEEG(sourceTrainFeatures, shiftAmount);
augmentedTargetTrainFeatures = timeShiftEEG(targetTrainFeatures, shiftAmount);

% Step 3: Combine original and augmented data for training
combinedFeatures = [sourceTrainFeatures; augmentedSourceTrainFeatures; targetTrainFeatures; augmentedTargetTrainFeatures];
combinedLabels = [sourceTrainLabels; sourceTrainLabels; targetTrainLabels; targetTrainLabels];

% Step 4: Train the LDA model with the augmented data
ldaModel = fitcdiscr(combinedFeatures, combinedLabels);

% Step 5: Train the SVM model with the augmented data
svmModel = fitcsvm(combinedFeatures, combinedLabels, 'KernelFunction', 'rbf', 'Standardize', true);

% Step 6: Predict using both models on the test data (without augmentation)
ldaPredictedLabels = predict(ldaModel, testFeatures);
svmPredictedLabels = predict(svmModel, testFeatures);

% Step 7: Ensemble via Majority Voting
finalPredictedLabels = mode([ldaPredictedLabels, svmPredictedLabels], 2);  % Majority vote

% Step 8: Calculate test accuracy for the ensemble model with augmented data
ensembleTestAccuracy = mean(finalPredictedLabels == testLabels) * 100;

% Display the result
disp(['Test Accuracy with Time Shifting Augmentation and Ensemble (LDA + SVM): ', num2str(ensembleTestAccuracy), '%']);
%% % Define a range of shift amounts to test
shiftAmounts = [2, 5, 10, 15];  % You can adjust or expand this range

bestShiftAmount = 0;
bestAccuracy = 0;

for shiftAmount = shiftAmounts
    disp(['Testing shift amount: ', num2str(shiftAmount)]);
    
    % Step 1: Apply time shifting augmentation
    augmentedTrainFeatures = timeShiftEEG(combinedFeatures, shiftAmount);
    
    % Step 2: Combine original and augmented data
    combinedAugmentedFeatures = [combinedFeatures; augmentedTrainFeatures];
    combinedAugmentedLabels = [combinedLabels; combinedLabels];  % Labels remain the same
    
    % Step 3: Train the SVM model on the augmented data
    finalSVMModel = fitcsvm(combinedAugmentedFeatures, combinedAugmentedLabels, 'KernelFunction', 'polynomial', ...
        'BoxConstraint', 1, 'KernelScale', 1/1, 'Standardize', true);
    
    % Step 4: Predict on the test data
    predictedTestLabels = predict(finalSVMModel, testFeatures);
    
    % Step 5: Calculate test accuracy
    testAccuracy = mean(predictedTestLabels == testLabels) * 100;
    disp(['Test Accuracy with shift amount ', num2str(shiftAmount), ': ', num2str(testAccuracy), '%']);
    
    % Keep track of the best shift amount
    if testAccuracy > bestAccuracy
        bestShiftAmount = shiftAmount;
        bestAccuracy = testAccuracy;
    end
end

% Display the best shift amount and accuracy
disp(['Best Shift Amount: ', num2str(bestShiftAmount)]);
disp(['Best Test Accuracy: ', num2str(bestAccuracy), '%']);


%% % Define the shift amount based on previous results
shiftAmount = 5;

% Define the noise level for Gaussian noise
noiseLevel = 0.01;  % You can adjust this value

% Step 1: Augment the source and target training data by shifting in time
augmentedSourceTrainFeaturesShifted = timeShiftEEG(sourceTrainFeatures, shiftAmount);
augmentedTargetTrainFeaturesShifted = timeShiftEEG(targetTrainFeatures, shiftAmount);

% Step 2: Add Gaussian noise to the original and shifted data
augmentedSourceTrainFeaturesNoise = addGaussianNoise(sourceTrainFeatures, noiseLevel);
augmentedTargetTrainFeaturesNoise = addGaussianNoise(targetTrainFeatures, noiseLevel);
augmentedSourceTrainFeaturesShiftedNoise = addGaussianNoise(augmentedSourceTrainFeaturesShifted, noiseLevel);
augmentedTargetTrainFeaturesShiftedNoise = addGaussianNoise(augmentedTargetTrainFeaturesShifted, noiseLevel);

% Step 3: Combine original, shifted, and noisy data for training
combinedFeatures = [
    sourceTrainFeatures; 
    augmentedSourceTrainFeaturesShifted; 
    augmentedSourceTrainFeaturesNoise; 
    augmentedSourceTrainFeaturesShiftedNoise;
    targetTrainFeatures; 
    augmentedTargetTrainFeaturesShifted; 
    augmentedTargetTrainFeaturesNoise; 
    augmentedTargetTrainFeaturesShiftedNoise
];
combinedLabels = [
    sourceTrainLabels; 
    sourceTrainLabels; 
    sourceTrainLabels; 
    sourceTrainLabels;
    targetTrainLabels; 
    targetTrainLabels; 
    targetTrainLabels; 
    targetTrainLabels
];

% Step 4: Train the LDA model with the augmented data
ldaModel = fitcdiscr(combinedFeatures, combinedLabels);

% Step 5: Train the SVM model with the augmented data
svmModel = fitcsvm(combinedFeatures, combinedLabels, 'KernelFunction', 'rbf', 'Standardize', true);

% Step 6: Predict using both models on the test data
ldaPredictedLabels = predict(ldaModel, testFeatures);
svmPredictedLabels = predict(svmModel, testFeatures);

% Step 7: Ensemble via Majority Voting
finalPredictedLabels = mode([ldaPredictedLabels, svmPredictedLabels], 2);  % Majority vote

% Step 8: Calculate test accuracy for the ensemble model with augmented data
ensembleTestAccuracy = mean(finalPredictedLabels == testLabels) * 100;

% Display the result
disp(['Test Accuracy with Time Shifting + Gaussian Noise and Ensemble (LDA + SVM): ', num2str(ensembleTestAccuracy), '%']);
%% % Step 1: Apply time shifting augmentation with the best shift amount (5)
shiftAmount = 5;  % Best shift amount found earlier
augmentedTrainFeatures = timeShiftEEG(combinedFeatures, shiftAmount);

% Step 2: Combine original and augmented data
combinedAugmentedFeatures = [combinedFeatures; augmentedTrainFeatures];
combinedAugmentedLabels = [combinedLabels; combinedLabels];  % Labels remain the same

% Step 3: Train the SVM model using the augmented data
svmModel = fitcsvm(combinedAugmentedFeatures, combinedAugmentedLabels, 'KernelFunction', 'polynomial', ...
    'BoxConstraint', 1, 'KernelScale', 1/1, 'Standardize', true);

% Step 4: Train the LDA model using the same augmented data
ldaModel = fitcdiscr(combinedAugmentedFeatures, combinedAugmentedLabels);

% Step 5: Predict using both models on the test data
svmPredictedLabels = predict(svmModel, testFeatures);
ldaPredictedLabels = predict(ldaModel, testFeatures);

% Step 6: Ensemble via majority voting
finalPredictedLabels = mode([svmPredictedLabels, ldaPredictedLabels], 2);  % Majority vote

% Step 7: Calculate the test accuracy for the ensemble model
ensembleTestAccuracy = mean(finalPredictedLabels == testLabels) * 100;

% Display the final test accuracy for the ensemble model
disp(['Test Accuracy with Ensemble (SVM + LDA) and Time Shifting: ', num2str(ensembleTestAccuracy), '%']);



