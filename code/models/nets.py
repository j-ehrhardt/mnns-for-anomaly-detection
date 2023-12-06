from .modules import *

############################## Monolithic ###################################################################

class MonolithAE(nn.Module):
    def __init__(self, hparam, in_ch, in_len):
        super(MonolithAE, self).__init__()

        self.encoder = AE_Encoder(hparam=hparam, in_=in_len)
        self.decoder = AE_Decoder(hparam=hparam, lat_=self.encoder.lat_, out_=in_len)


    def forward(self, x):
        x_f = torch.flatten(x, start_dim=1)
        lat = self.encoder(x_f)
        dec = self.decoder(lat)
        out = torch.reshape(dec, x.shape)
        return out


class MonolithVAE(nn.Module):
    def __init__(self, hparam, in_ch, in_len):
        super(MonolithVAE, self).__init__()

        self.encoder = VAE_Encoder(hparam=hparam, in_=in_len)
        self.decoder = VAE_Decoder(hparam=hparam, lat_=self.encoder.lat_, out_=in_len)

    def forward(self, x):
        x_f = torch.flatten(x, start_dim=1)
        lat = self.encoder(x_f)
        dec = self.decoder(lat)
        out = torch.reshape(dec, x.shape)
        return out


############################## Modular Type 0 - Agnostic Type ###############################################################

class Modular0AE(nn.Module):
    def __init__(self, hparam, in_ch, in_len):
        super(Modular0AE, self).__init__()
        self.in_ch = in_ch
        mod_in_len = int(in_len / in_ch)

        AEs = []
        for i in range(in_ch):
            AEs.append(MonolithAE(hparam=hparam, in_ch=1, in_len=mod_in_len))
        self.AEs = nn.ModuleList(AEs)

    def forward(self, x):
        for i in range(self.in_ch):
            out = self.AEs[i](x[:, :, i])
            x_out = out.unsqueeze(dim=2) if i == 0 else torch.cat((x_out, out.unsqueeze(dim=2)), dim=2)
        return x_out


class Modular0VAE(nn.Module):
    def __init__(self, hparam, in_ch, in_len):
        super(Modular0VAE, self).__init__()
        self.in_ch = in_ch
        mod_in_len = int(in_len / in_ch)

        AEs = []
        for i in range(in_ch):
            AEs.append(MonolithVAE(hparam=hparam, in_ch=1, in_len=mod_in_len))
        self.AEs = nn.ModuleList(AEs)


    def forward(self, x):
        for i in range(self.in_ch):
            out = self.AEs[i](x[:, :, i])
            x_out = out.unsqueeze(dim=2) if i == 0 else torch.cat((x_out, out.unsqueeze(dim=2)), dim=2)
        return x_out



############################## Modular Type 1 - Divergent Type ##############################################################

class Modular1AE(nn.Module):
    def __init__(self, hparam, in_ch, in_len):
        super(Modular1AE, self).__init__()
        self.in_ch = in_ch
        mod_in_len = int(in_len / in_ch)

        AEs = []
        for i in range(in_ch):
            AEs.append(MonolithAE(hparam=hparam, in_ch=1, in_len=mod_in_len))
        self.AEs = nn.ModuleList(AEs)

    def forward(self, x):
        x_med = self.AEs[0](x[:, :, 0])
        x_out = x_med.unsqueeze(dim=2)

        for i in range(1, self.in_ch):
            out = self.AEs[i](x_med)
            x_out = torch.cat((x_out, out.unsqueeze(dim=2)), dim=2)
        return x_out


class Modular1VAE(nn.Module):
    def __init__(self, hparam, in_ch, in_len):
        super(Modular1VAE, self).__init__()
        self.in_ch = in_ch
        mod_in_len = int(in_len / in_ch)

        AEs = []
        for i in range(in_ch):
            AEs.append(MonolithVAE(hparam=hparam, in_ch=1, in_len=mod_in_len))
        self.AEs = nn.ModuleList(AEs)

    def forward(self, x):
        x_med = self.AEs[0](x[:, :, 0])
        x_out = x_med.unsqueeze(dim=2)

        for i in range(1, self.in_ch):
            out = self.AEs[i](x_med)
            x_out = torch.cat((x_out, out.unsqueeze(dim=2)), dim=2)
        return x_out


############################## Modular Type III - Sequential Type #############################################################

class Modular2AE(nn.Module):
    def __init__(self, hparam, in_ch, in_len):
        super(Modular2AE, self).__init__()
        self.in_ch = in_ch
        mod_in_len = int(in_len / in_ch)

        AEs = []
        for i in range(in_ch):
            AEs.append(MonolithAE(hparam=hparam, in_ch=1, in_len=mod_in_len))
        self.AEs = nn.ModuleList(AEs)

    def forward(self, x):
        for i in range(self.in_ch):
            out = self.AEs[0](x[:, :, 0:1]) if i == 0 else self.AEs[i](out)
            x_out = out if i == 0 else torch.cat((x_out, out), dim=2)
        return x_out


class Modular2VAE(nn.Module):
    def __init__(self, hparam, in_ch, in_len):
        super(Modular2VAE, self).__init__()
        self.in_ch = in_ch
        mod_in_len = int(in_len / in_ch)

        AEs = []
        for i in range(in_ch):
            AEs.append(MonolithVAE(hparam=hparam, in_ch=1, in_len=mod_in_len))
        self.AEs = nn.ModuleList(AEs)

    def forward(self, x):
        for i in range(self.in_ch):
            out = self.AEs[0](x[:, :, 0:1]) if i == 0 else self.AEs[i](out)
            x_out = out if i == 0 else torch.cat((x_out, out), dim=2)
        return x_out