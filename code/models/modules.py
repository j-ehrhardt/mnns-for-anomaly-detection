import torch
import torch.nn as nn

class AE_Encoder(nn.Module):
    def __init__(self, hparam, in_):
        super(AE_Encoder, self).__init__()
        self.in_ = in_
        self.act = torch.nn.SELU()
        self.dropout = nn.Dropout(p=hparam["DROPOUT_P"], inplace=True)

        layers = []
        for i in range(hparam["N_LAYERS"]):
            in_ = in_ if i == 0 else int(in_ / hparam["AE_F_ENCODING"])
            enc_ = int(in_ / hparam["AE_F_ENCODING"])
            layers += [nn.Linear(in_features=int(in_), out_features=int(enc_), bias=True), self.act, self.dropout]
        self.encoder = nn.Sequential(*layers)
        self.lat_ = enc_

    def forward(self, x):
        z = self.encoder(x)
        return z

    def get_dims(self):
        return self.in_, self.lat_


class AE_Decoder(nn.Module):
    def __init__(self, hparam, lat_, out_):
        super(AE_Decoder, self).__init__()

        self.act = torch.nn.SELU()
        self.dropout = nn.Dropout(p=hparam["DROPOUT_P"])

        layers = []
        for i in range(hparam["N_LAYERS"]):
            lat_ = lat_ if i == 0 else int(lat_ * hparam["AE_F_ENCODING"])
            dec_ = out_ if i == hparam["N_LAYERS"] -1 else int(lat_ * hparam["AE_F_ENCODING"])
            layers += [nn.Linear(in_features=lat_, out_features=dec_, bias=True), self.act, self.dropout] if i != hparam["N_LAYERS"] - 1 else [nn.Linear(in_features=lat_, out_features=dec_, bias=True)]
        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        y = self.decoder(x)
        return y


class VAE_Encoder(nn.Module):
    def __init__(self, hparam, in_):
        super(VAE_Encoder, self).__init__()
        self.in_ = in_
        self.act = torch.nn.SELU()
        self.dropout = nn.Dropout(p=hparam["DROPOUT_P"], inplace=True)

        self.N = torch.distributions.Normal(0,1)
        self.N.loc = self.N.loc.cuda()  # get sampling on GPU
        self.N.scale = self.N.scale.cuda()
        self.kl = 0

        layers = []
        for i in range(hparam["N_LAYERS"]):
            in_ = in_ if i ==0 else int(in_ / hparam["AE_F_ENCODING"])
            enc_ = int(in_ / hparam["AE_F_ENCODING"])
            layers += [nn.Linear(in_features=int(in_), out_features=int(enc_), bias=True), self.act, self.dropout]

        self.sampling1 = nn.Linear(in_features=int(enc_), out_features=int(enc_), bias=True)
        self.sampling2 = nn.Linear(in_features=int(enc_), out_features=int(enc_), bias=True)

        self.encoder = nn.Sequential(*layers)
        self.lat_ = enc_

    def forward(self, x):
        x = self.encoder(x)

        mu = self.sampling1(x)
        sigma = torch.exp(self.sampling2(x))
        z = mu + sigma * self.N.sample(mu.shape)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1 / 2).sum()
        return z


class VAE_Decoder(nn.Module):
    def __init__(self, hparam, lat_, out_):
        super(VAE_Decoder, self).__init__()

        self.act = torch.nn.SELU()
        self.dropout = nn.Dropout(p=hparam["DROPOUT_P"])

        layers = []
        for i in range(hparam["N_LAYERS"]):
            lat_ = int(lat_) if i == 0 else int(lat_ * hparam["AE_F_ENCODING"])
            dec_ = out_ if i == hparam["N_LAYERS"] -1 else int(lat_ * hparam["AE_F_ENCODING"])
            layers += [nn.Linear(in_features=lat_, out_features=dec_, bias=True), self.act, self.dropout] if i != hparam["N_LAYERS"] - 1 else [nn.Linear(in_features=lat_, out_features=dec_, bias=True)]
        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        y = self.decoder(x)
        return y

