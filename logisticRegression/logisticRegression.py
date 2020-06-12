from sklearn import linear_model

x = [[20, 3],
     [23, 7],
     [31, 10],
     [42, 13],
     [50, 7],
     [60, 5]]

y = [0,
     1,
     1,
     1,
     0,
     0]

lr = linear_model.LogisticRegression()
lr.fit(x, y)
#
testX = [[28, 8]]


label = lr.predict(testX)      # predicted logistic regression of y
print('predicted label = ', label)

prob = lr.predict_proba(testX)        # predicted probability of [P(y=0), P(y=1)]
print('The probability of 0/1 is', prob)

# %%
theta_0 = lr.intercept_
theta_1 = lr.coef_[0][0]
theta_2 = lr.coef_[0][1]

print('theta_0 = ', theta_0)
print('theta_1 = ', theta_1)
print('theta_2 = ', theta_2)

testX = [[28, 8]]
ratio = prob[0][1]/prob[0][0]

testX = [[28, 9]]
prob_new = lr.predict_proba(testX)
ratio_new = prob_new[0][1]/prob_new[0][0]

ratio_of_radio = ratio_new / ratio
print('ratio_of_radio = ', ratio_of_radio)

import math
theta_2_e = math.exp(theta_2)
print('theta 2 e =', theta_2_e)


