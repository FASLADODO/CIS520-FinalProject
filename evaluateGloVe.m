function [ y_hat ] = evaluateGloVe( X, vecs, happy, sad )
%EVALUATEGLOVE Summary of this function goes here
%   Detailed explanation goes here
y_hat = zeros(size(X, 1), 1);

for i = 1:size(X, 1)
    currentScore = 0;
    for j = find(X(i, :))
        happy_sim = dot(vecs(j, :), happy);
        sad_sim = dot(vecs(j, :), sad);
        
        % Choose the one more similar
        if happy_sim > sad_sim
            currentScore = currentScore + X(i, j)*happy_sim;
        else
            currentScore = currentScore - X(i, j)*sad_sim;
        end
    end
    
    % >= slightly biases towards happy, which is fine, because so is the
    % data
    y_hat = currentScore >= 0;
end
end

