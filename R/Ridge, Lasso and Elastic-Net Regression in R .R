library(glmnet)

set.seed(42)

n <- 1000
p <- 5000
real_p <- 15

x <- matrix(rnorm(n*p), nrow = n, ncol = p)
y <- apply(x[,1:real_p],1,sum) + rnorm(n)

#sample will randomly picks values 1 to n
# .66*n 66% row will be in training set
train_rows <-sample(1:n, .66*n)

x.train <- x[train_rows,]
x.test <- x[-train_rows,]


y.train <- y[train_rows]
y.test <- y[-train_rows]

# alpha=0 represents ridge
# family='gaussian' denotes linear
# for logestic regression >> family='binomial'
alpha0.fit <- cv.glmnet(x.train,y.train, type.measure = 'mse', alpha=0, family='gaussian')
alpha0.predict <- predict(alpha0.fit,s=alpha0.fit$lambda.1se, newx = x.test)
mean((y.test - alpha0.predict)^2)

# alpha=0 represents lasso
alpha1.fit <- cv.glmnet(x.train,y.train, type.measure = 'mse', alpha=1, family='gaussian')
alpha1.predict <- predict(alpha1.fit,s=alpha1.fit$lambda.1se, newx = x.test)
mean((y.test - alpha1.predict)^2)

# alpha= anything between 0-1 represents elasticnet
alpha0.5.fit <- cv.glmnet(x.train,y.train, type.measure = 'mse', alpha=0.5, family='gaussian')
alpha0.5.predict <- predict(alpha0.5.fit,s=alpha0.5.fit$lambda.1se, newx = x.test)
mean((y.test - alpha0.5.predict)^2)
