
## BART: Bayesian Additive Regression Trees
## Copyright (C) 2017 Robert McCulloch and Rodney Sparapani

## This program is free software; you can redistribute it and/or modify
## it under the terms of the GNU General Public License as published by
## the Free Software Foundation; either version 2 of the License, or
## (at your option) any later version.

## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.

## You should have received a copy of the GNU General Public License
## along with this program; if not, a copy is available at
## https://www.R-project.org/Licenses/GPL-2

predict_prob_b <- function(object, newdata, b = NULL, mc.cores=1, 
                            openmp=(mc.cores.openmp()>0), ...) {
  
    stopifnot(class(object) == "pbart")
    if (is.null(b)) {
      stop('Need to specify which posterior draw to do prediction for!')    
    }

    p <- length(object$treedraws$cutpoints)

    if(p!=ncol(newdata))
        stop(paste0('The number of columns in newdata must be equal to ', p))

    if(length(object$binaryOffset)==0) object$binaryOffset=object$offset

    yhat.test <- pwbart_b(b = b, newdata, object$treedraws, mc.cores=1,
                                    mu=object$binaryOffset, ...)

    prob.test <- pnorm(yhat.test)
    return(prob.test)
}

