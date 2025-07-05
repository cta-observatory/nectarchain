try:
    import sys

    import numpy as np
    from IPython import embed
    from scipy.stats import norm, poisson

except ImportError as e:
    print(e)
    raise SystemExit


class GaussHistoFitFunction:
    def __init__(
        self, bin_edges, bin_contents, integrate=True
    ):  # Assume that the bin centers are "integers" and that they are spaced at 1.... refinement will be for later
        self.xcentre = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        self.xedge = bin_edges
        self.y = bin_contents
        self.use_integration = integrate

    def ExpectedValues(self, normalisation, mu, sigma):
        y = (
            normalisation
            * np.exp(-0.5 * np.power((self.xcentre - mu) / sigma, 2.0))
            / (np.sqrt(2.0 * np.pi) * sigma)
        )
        return y

    def IntegratedExpectedValues(self, normalisation, mu, sigma):
        yup = norm.cdf(self.xedge[1:], loc=mu, scale=sigma)
        ylow = norm.cdf(self.xedge[:-1], loc=mu, scale=sigma)
        return normalisation * (yup - ylow)

    def Minus2LogLikelihood(self, parameters):
        normalisation = parameters[0]
        mu = parameters[1]
        sigma = parameters[2]
        if self.use_integration:
            y_fit = self.IntegratedExpectedValues(normalisation, mu, sigma)
        else:
            y_fit = self.ExpectedValues(normalisation, mu, sigma)
        # print(y_fit)
        log_likes = poisson.logpmf(self.y, y_fit)
        total = np.sum(log_likes)
        return -2.0 * total


class JPT2FitFunction:
    def __init__(self, bin_edges, bin_contents):
        self.xcentre = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        self.xedge = bin_edges
        self.y = bin_contents.astype(float)

    def Prediction(self, parameters):
        amplitude = parameters[0]
        hiPed = parameters[1]
        sigmaHiPed = parameters[2]
        ns = parameters[3]
        meanillu = parameters[4]
        gain = parameters[5]
        sigmaGain = parameters[6]

        ## Compute the maximum illumination to consider
        nMuUsed = int(5.0 * np.sqrt(meanillu + 1)) + 2

        ## Precompute some parameters that will be needed for the full iteration
        imu = np.arange(nMuUsed)
        sig2 = imu * (sigmaGain * sigmaGain) + sigmaHiPed * sigmaHiPed
        pos_x = imu * gain + hiPed
        poiG = poisson.pmf(imu, meanillu) / np.sqrt(2.0 * np.pi * sig2)
        poiG[1:] *= ns

        sig2x2 = 2.0 * sig2
        ## Loop on pixels:

        predicted_entries = np.zeros_like(self.y)

        # embed()
        for i in range(self.xcentre.shape[0]):
            predicted_entries[i] = np.sum(
                poiG * np.exp(-np.power(self.xcentre[i] - pos_x, 2.0) / sig2x2)
            )

        predicted_entries *= amplitude
        return predicted_entries

    def Minus2LogLikelihood(self, parameters):
        # print(f'Minuit2 Called {parameters}')
        y_pred = self.Prediction(parameters)

        LogLikeBins = poisson.logpmf(self.y, y_pred)
        LogLike = np.sum(LogLikeBins)
        # if np.isnan(LogLike):
        #    print("--> Nan")
        Minus2LogLike = -2.0 * LogLike
        return Minus2LogLike
