library(MASS)
k = 1000000
mean1 = c(4,2,6,-1,-4)
mean2 = c(-4,2,1,2,6)

var1 = 3
var2 = 2

n1 = mvrnorm(n=k,mean1,diag(var1,5))
n2 = mvrnorm(n=k,mean2,diag(var2,5))

diff = n1-n2

diff_norm = rep(0,k)
for (i in 1:k){
  diff_norm[i] = norm(diff[i,],'2')
  if (i %% 100000 == 0){
    print(i)
  }
}




# Non-centered scaled chi-sq
ncp = (var1+var2)^(-1)*t(mean1-mean2)%*%(mean1-mean2)
diff_norm_ana = sqrt((var1+var2)*rchisq(k, 5, ncp = ncp))


mean(diff_norm)
var(diff_norm)
median(diff_norm)


mean(diff_norm_ana)
var(diff_norm_ana)
median(diff_norm_ana)



