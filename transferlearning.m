% Define the source and target subjects
sourceSubjectIndex = 1; % You can change this to another source subject
targetSubjectIndex = 2; % You can change this to another target subject

%% Step 1: Train LDA model on the source subject (same as before)
% Load and preprocess the source subject data
[Raw_sub, EMap] = loadbci4eegimagery([subjects{sourceSubjectIndex} '.mat'], [769 770]);
xsubi = extracteegbci4imagery(Raw_sub, 'indicate', 'seconds', [0 3], 'selchs', EMap(1, subjectsi{sourceSubjectIndex}));
clear Raw_sub;

% Apply filtering and feature extraction
finaltrn1.x = filter(b1, a1, xsubi.x);
finaltrn.x = finaltrn1.x(126:625, :, :);
finaltrn.y = xsubi.y;

% Extract features using CSP for the source subject
numchannel = 22;
[finalfeaturtrn, selectedw1] = featcrossval(finaltrn, 1:22, numchannel);

% Prepare the source subject data
sourceTrainFeatures = finalfeaturtrn.x';
sourceTrainLabels = finalfeaturtrn.y - 1;

% Train the LDA model on the source subject
sourceLDA_Model = fitcdiscr(sourceTrainFeatures, sourceTrainLabels);

%% Step 2: Apply CORAL to align the source features to the target domain
% Load and preprocess the target subject data
[Raw_sub, EMap] = loadbci4eegimagery([subjects{targetSubjectIndex} '.mat'], [769 770]);
xsubi = extracteegbci4imagery(Raw_sub, 'indicate', 'seconds', [0 3], 'selchs', EMap(1, subjectsi{targetSubjectIndex}));
clear Raw_sub;

% Apply filtering and feature extraction
finaltrn1.x = filter(b1, a1, xsubi.x);
finaltrn.x = finaltrn1.x(126:625, :, :);
finaltrn.y = xsubi.y;

% Extract features using CSP for the target subject
[finalfeaturtrn, ~] = featcrossval(finaltrn, 1:22, numchannel);

% Prepare the target subject data for fine-tuning
targetTrainFeatures = finalfeaturtrn.x';
targetTrainLabels = finalfeaturtrn.y - 1;

% Apply CORAL: Align source features to target domain
C_source = cov(sourceTrainFeatures);
C_target = cov(targetTrainFeatures);
reg = 1e-6;
C_source = C_source + reg * eye(size(C_source)); % Regularization
C_target = C_target + reg * eye(size(C_target)); % Regularization
sourceAligned = sourceTrainFeatures * (C_target^(0.5)) / (C_source^(0.5));

% Combine aligned source data with the target data
combinedFeatures = [sourceAligned; targetTrainFeatures];
combinedLabels = [sourceTrainLabels; targetTrainLabels];

%% Step 3: Fine-tune the LDA model with the aligned source and target data
% Fine-tune the LDA model with combined (aligned) data
updatedLDA_Model = fitcdiscr(combinedFeatures, combinedLabels);

%% Step 4: Test the fine-tuned model on target subject's test data
% Use your existing test data for the target subject (testFeatures and testLabels)
predictedTestLabels = predict(updatedLDA_Model, testFeatures);
finalTestAccuracy = mean(predictedTestLabels == testLabels) * 100;

% Display the result
disp(['Test Accuracy after CORAL and Fine-Tuning on Target Subject: ' num2str(finalTestAccuracy) '%']);
