library(fitdistrplus)
library(logspline)
data_df <- read.csv(file = 'C:\\Users\\ssbeh\\Downloads\\OAI_analysis\\plots\\Nvidia RTX 20803.csv')
data <- data_df[[1]]


#descdist(data, discrete = FALSE)
#fw <- fitdist(data, "pois")


fg <- fitdist(data, "gamma")


flg <- fitdist(data, "lnorm")
par(mfrow = c(2, 2))
plot.legend <- c("dgamma", "lnorm")
denscomp(list(fg, flg), legendtext = plot.legend)
qqcomp(list(fg, flg), legendtext = plot.legend)
cdfcomp(list(fg, flg), legendtext = plot.legend)
ppcomp(list(fg, flg), legendtext = plot.legend)
