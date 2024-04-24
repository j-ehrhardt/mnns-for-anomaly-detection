from .modules import *

############################## Monolithic ###################################################################

class MonolithAE(nn.Module):
    def __init__(self, hparam, in_ch, out_ch, window_size):
        super(MonolithAE, self).__init__()

        self.encoder = AE_Encoder(hparam=hparam, in_=window_size*in_ch)
        self.decoder = AE_Decoder(hparam=hparam, lat_=self.encoder.lat_, out_=window_size*out_ch)


    def forward(self, x, y):
        x_f = torch.flatten(x, start_dim=1)
        lat = self.encoder(x_f)
        dec = self.decoder(lat)
        out = torch.reshape(dec, y.shape)
        return out


class MonolithVAE(nn.Module):
    def __init__(self, hparam, in_ch, out_ch, window_size):
        super(MonolithVAE, self).__init__()

        self.encoder = VAE_Encoder(hparam=hparam, in_=window_size*in_ch)
        self.decoder = VAE_Decoder(hparam=hparam, lat_=self.encoder.lat_, out_=window_size*out_ch)

    def forward(self, x, y):
        x_f = torch.flatten(x, start_dim=1)
        lat = self.encoder(x_f)
        dec = self.decoder(lat)
        out = torch.reshape(dec, y.shape)
        return out


class MonolithRNN(nn.Module):
    def __init__(self, hparam, in_ch, out_ch, window_size):
        super(MonolithRNN, self).__init__()

        latent_size = hparam["LATENT_SIZE"]
        hidden_sizes = hparam["HIDDEN_SIZES"]

        self.encoder = RNN_Encoder(in_ch, latent_size, hidden_sizes)
        self.decoder = RNN_Decoder(latent_size, out_ch, hidden_sizes[::-1])

    def forward(self, x, y):
        seq_len = x.shape[1]
        z = self.encoder(x)
        y_hat = self.decoder(z, seq_len)
        return torch.reshape(y_hat, y.shape)


class MonolithLSTM(nn.Module):
    def __init__(self, hparam, in_ch, out_ch, window_size):
        super(MonolithLSTM, self).__init__()

        latent_size = hparam["LATENT_SIZE"]
        hidden_sizes = hparam["HIDDEN_SIZES"]

        self.encoder = LSTM_Encoder(in_ch, latent_size, hidden_sizes)
        self.decoder = LSTM_Decoder(latent_size, out_ch, hidden_sizes[::-1])


    def forward(self, x, y):
        seq_len = x.shape[1]
        z = self.encoder(x)
        y_hat = self.decoder(z, seq_len)
        return torch.reshape(y_hat, y.shape)


class MonolithGRU(nn.Module):
    def __init__(self, hparam, in_ch, out_ch, window_size):
        super(MonolithGRU, self).__init__()

        latent_size = hparam["LATENT_SIZE"]
        hidden_sizes = hparam["HIDDEN_SIZES"]

        self.encoder = GRU_Encoder(in_ch, latent_size, hidden_sizes)
        self.decoder = GRU_Decoder(latent_size, out_ch, hidden_sizes[::-1])

    def forward(self, x, y):
        seq_len = x.shape[1]
        z = self.encoder(x)
        y_hat = self.decoder(z, seq_len)
        return torch.reshape(y_hat, y.shape)


############################## Modular Type 0 - Agnostic Type ###############################################################

class Modular0AE(nn.Module):
    def __init__(self, hparam, in_ch, out_ch, window_size):
        super(Modular0AE, self).__init__()
        self.out_ch = out_ch

        AEs = []
        for i in range(out_ch):
            AEs.append(MonolithAE(hparam=hparam, in_ch=2, out_ch=1, window_size=window_size))
        self.AEs = nn.ModuleList(AEs)

    def forward(self, x, y):
        for i in range(self.out_ch):
            x_in = torch.cat((x[:, :, i].unsqueeze(dim=2), x[:, :, int(i + self.out_ch -1)].unsqueeze(dim=2)), dim=2)
            y_hat_i = self.AEs[i](x_in, y[:, :, i].unsqueeze(dim=2))
            y_hat = y_hat_i if i == 0 else torch.cat((y_hat, y_hat_i), dim=2)
        return y_hat


class Modular0VAE(nn.Module):
    def __init__(self, hparam, in_ch, out_ch, window_size):
        super(Modular0VAE, self).__init__()
        self.out_ch = out_ch

        AEs = []
        for i in range(out_ch):
            AEs.append(MonolithVAE(hparam=hparam, in_ch=2, out_ch=1, window_size=window_size))
        self.AEs = nn.ModuleList(AEs)


    def forward(self, x, y):
        for i in range(self.out_ch):
            x_in = torch.cat((x[:, :, i].unsqueeze(dim=2), x[:, :, int(i + self.out_ch -1)].unsqueeze(dim=2)), dim=2)
            y_hat_i = self.AEs[i](x_in, y[:, :, i].unsqueeze(dim=2))
            y_hat = y_hat_i if i == 0 else torch.cat((y_hat, y_hat_i), dim=2)
        return y_hat


class Modular0RNN(nn.Module):
    def __init__(self, hparam, in_ch, out_ch, window_size):
        super(Modular0RNN, self).__init__()
        self.out_ch = out_ch

        RNNs = []
        for i in range(out_ch):
            RNNs.append(MonolithRNN(hparam=hparam, in_ch=2, out_ch=1, window_size=window_size))
        self.RNNs = nn.ModuleList(RNNs)

    def forward(self, x, y):
        for i in range(self.out_ch):
            x_in = torch.cat((x[:, :, i].unsqueeze(dim=2), x[:, :, int(i + self.out_ch -1)].unsqueeze(dim=2)), dim=2)
            out = self.RNNs[i](x_in, y[:, :, i].unsqueeze(dim=2))
            y_out = out if i == 0 else torch.cat((y_out, out), dim=2)
        return y_out


class Modular0LSTM(nn.Module):
    def __init__(self, hparam, in_ch, out_ch, window_size):
        super(Modular0LSTM, self).__init__()
        self.out_ch = out_ch

        RNNs = []
        for i in range(out_ch):
            RNNs.append(MonolithLSTM(hparam=hparam, in_ch=2, out_ch=1, window_size=window_size))
        self.RNNs = nn.ModuleList(RNNs)

    def forward(self, x, y):
        for i in range(self.out_ch):
            x_in = torch.cat((x[:, :, i].unsqueeze(dim=2), x[:, :, int(i + self.out_ch -1)].unsqueeze(dim=2)), dim=2)
            out = self.RNNs[i](x_in, y[:, :, i].unsqueeze(dim=2))
            y_out = out if i == 0 else torch.cat((y_out, out), dim=2)
        return y_out


class Modular0GRU(nn.Module):
    def __init__(self, hparam, in_ch, out_ch, window_size):
        super(Modular0GRU, self).__init__()
        self.out_ch = out_ch

        RNNs = []
        for i in range(out_ch):
            RNNs.append(MonolithGRU(hparam=hparam, in_ch=2, out_ch=1, window_size=window_size))
        self.RNNs = nn.ModuleList(RNNs)

    def forward(self, x, y):
        for i in range(self.out_ch):
            x_in = torch.cat((x[:, :, i].unsqueeze(dim=2), x[:, :, int(i + self.out_ch -1)].unsqueeze(dim=2)), dim=2)
            out = self.RNNs[i](x_in, y[:, :, i].unsqueeze(dim=2))
            y_out = out if i == 0 else torch.cat((y_out, out), dim=2)
        return y_out

############################## Modular Type 1 - Divergent Type ##############################################################

class Modular1AE(nn.Module):
    def __init__(self, hparam, in_ch, out_ch, window_size):
        super(Modular1AE, self).__init__()
        self.out_ch = out_ch

        AEs = []
        for i in range(out_ch):
            AEs.append(MonolithAE(hparam=hparam, in_ch=2, out_ch=1, window_size=window_size)) if i == 0 \
                else AEs.append(MonolithAE(hparam=hparam, in_ch=3, out_ch=1, window_size=window_size))
        self.AEs = nn.ModuleList(AEs)

    def forward(self, x, y):
        out_0 = self.AEs[0](torch.cat((x[:,:,0].unsqueeze(dim=2), x[:,:,5].unsqueeze(dim=2)), dim=2), y[:,:,0].unsqueeze(dim=2))
        y_out = out_0

        for i in range(1, self.out_ch):
            x_in = torch.cat((x[:,:,i].unsqueeze(dim=2), x[:,:,int(i+self.out_ch-1)].unsqueeze(dim=2), out_0), dim=2)
            out = self.AEs[i](x_in, y[:,:,i].unsqueeze(dim=2))
            y_out = torch.cat((y_out, out), dim=2)
        return y_out


class Modular1VAE(nn.Module):
    def __init__(self, hparam, in_ch, out_ch, window_size):
        super(Modular1VAE, self).__init__()
        self.out_ch = out_ch

        AEs = []
        for i in range(out_ch):
            AEs.append(MonolithVAE(hparam=hparam, in_ch=2, out_ch=1, window_size=window_size)) if i == 0 \
                else AEs.append(MonolithVAE(hparam=hparam, in_ch=3, out_ch=1, window_size=window_size))
        self.AEs = nn.ModuleList(AEs)

    def forward(self, x, y):
        out_0 = self.AEs[0](torch.cat((x[:,:,0].unsqueeze(dim=2), x[:,:,5].unsqueeze(dim=2)), dim=2), y[:,:,0].unsqueeze(dim=2))
        y_out = out_0

        for i in range(1, self.out_ch):
            x_in = torch.cat((x[:,:,i].unsqueeze(dim=2), x[:,:,int(i+self.out_ch-1)].unsqueeze(dim=2), out_0), dim=2)
            out = self.AEs[i](x_in, y[:,:,i].unsqueeze(dim=2))
            y_out = torch.cat((y_out, out), dim=2)
        return y_out


class Modular1RNN(nn.Module):
    def __init__(self, hparam, in_ch, out_ch, window_size):
        super(Modular1RNN, self).__init__()
        self.out_ch = out_ch

        RNNs = []
        for i in range(in_ch):
            RNNs.append(MonolithRNN(hparam=hparam, in_ch=2, out_ch=1, window_size=window_size)) if i == 0 \
                else RNNs.append(MonolithRNN(hparam=hparam, in_ch=3, out_ch=1, window_size=window_size))
        self.RNNs = nn.ModuleList(RNNs)

    def forward(self, x, y):
        out_0 = self.RNNs[0](torch.cat((x[:,:,0].unsqueeze(dim=2), x[:,:,5].unsqueeze(dim=2)), dim=2), y[:,:,0].unsqueeze(dim=2))
        y_out = out_0

        for i in range(1, self.out_ch):
            x_in = torch.cat((x[:,:,i].unsqueeze(dim=2), x[:,:,int(i+self.out_ch-1)].unsqueeze(dim=2), out_0), dim=2)
            out = self.RNNs[i](x_in, y[:,:,i].unsqueeze(dim=2))
            y_out = torch.cat((y_out, out), dim=2)
        return y_out


class Modular1LSTM(nn.Module):
    def __init__(self, hparam, in_ch, out_ch, window_size):
        super(Modular1LSTM, self).__init__()
        self.out_ch = out_ch

        RNNs = []
        for i in range(in_ch):
            RNNs.append(MonolithLSTM(hparam=hparam, in_ch=2, out_ch=1, window_size=window_size)) if i == 0 \
                else RNNs.append(MonolithLSTM(hparam=hparam, in_ch=3, out_ch=1, window_size=window_size))
        self.RNNs = nn.ModuleList(RNNs)

    def forward(self, x, y):
        out_0 = self.RNNs[0](torch.cat((x[:,:,0].unsqueeze(dim=2), x[:,:,5].unsqueeze(dim=2)), dim=2), y[:,:,0].unsqueeze(dim=2))
        y_out = out_0

        for i in range(1, self.out_ch):
            x_in = torch.cat((x[:,:,i].unsqueeze(dim=2), x[:,:,int(i+self.out_ch-1)].unsqueeze(dim=2), out_0), dim=2)
            out = self.RNNs[i](x_in, y[:,:,i].unsqueeze(dim=2))
            y_out = torch.cat((y_out, out), dim=2)
        return y_out


class Modular1GRU(nn.Module):
    def __init__(self, hparam, in_ch, out_ch, window_size):
        super(Modular1GRU, self).__init__()
        self.out_ch = out_ch

        RNNs = []
        for i in range(in_ch):
            RNNs.append(MonolithGRU(hparam=hparam, in_ch=2, out_ch=1, window_size=window_size)) if i == 0 \
                else RNNs.append(MonolithGRU(hparam=hparam, in_ch=3, out_ch=1, window_size=window_size))
        self.RNNs = nn.ModuleList(RNNs)

    def forward(self, x, y):
        out_0 = self.RNNs[0](torch.cat((x[:,:,0].unsqueeze(dim=2), x[:,:,5].unsqueeze(dim=2)), dim=2), y[:,:,0].unsqueeze(dim=2))
        y_out = out_0

        for i in range(1, self.out_ch):
            x_in = torch.cat((x[:,:,i].unsqueeze(dim=2), x[:,:,int(i+self.out_ch-1)].unsqueeze(dim=2), out_0), dim=2)
            out = self.RNNs[i](x_in, y[:,:,i].unsqueeze(dim=2))
            y_out = torch.cat((y_out, out), dim=2)
        return y_out

############################## Modular Type II - Convergent Type #############################################################

class Modular2AE(nn.Module):
    def __init__(self, hparam, in_ch, out_ch, window_size):
        super(Modular2AE, self).__init__()
        self.out_ch = out_ch

        AEs = []
        for i in range(out_ch-1):
            AEs.append(MonolithAE(hparam=hparam, in_ch=2, out_ch=1, window_size=window_size))
        AEs.append(MonolithAE(hparam=hparam, in_ch=7, out_ch=1, window_size=window_size))
        self.AEs = nn.ModuleList(AEs)

    def forward(self, x, y):
        for i in range(0, self.out_ch-1):
            x_in = torch.cat((x[:,:,i].unsqueeze(dim=2), x[:,:,int(i+self.out_ch-1)].unsqueeze(dim=2)), dim=2)
            out = self.AEs[i](x_in, y[:,:,i].unsqueeze(dim=2))
            y_out = out if i == 0 else torch.cat((y_out, out), dim=2)

        a = self.out_ch-1
        b = self.out_ch

        x_in = torch.cat((x[:,:,self.out_ch-1].unsqueeze(dim=2), x[:,:,-1].unsqueeze(dim=2), y_out), dim=2)
        out = self.AEs[self.out_ch-1](x_in, y[:,:,self.out_ch-1].unsqueeze(dim=2))
        y_out = torch.cat((y_out, out), dim=2)
        return y_out


class Modular2VAE(nn.Module):
    def __init__(self, hparam, in_ch, out_ch, window_size):
        super(Modular2VAE, self).__init__()
        self.out_ch = out_ch

        AEs = []
        for i in range(out_ch - 1):
            AEs.append(MonolithVAE(hparam=hparam, in_ch=2, out_ch=1, window_size=window_size))
        AEs.append(MonolithVAE(hparam=hparam, in_ch=7, out_ch=1, window_size=window_size))
        self.AEs = nn.ModuleList(AEs)

    def forward(self, x, y):
        for i in range(0, self.out_ch-1):
            x_in = torch.cat((x[:,:,i].unsqueeze(dim=2), x[:,:,int(i+self.out_ch-1)].unsqueeze(dim=2)), dim=2)
            out = self.AEs[i](x_in, y[:,:,i].unsqueeze(dim=2))
            y_out = out if i == 0 else torch.cat((y_out, out), dim=2)

        x_in = torch.cat((x[:,:,self.out_ch-1].unsqueeze(dim=2), x[:,:,-1].unsqueeze(dim=2), y_out), dim=2)
        out = self.AEs[self.out_ch-1](x_in, y[:,:,self.out_ch-1].unsqueeze(dim=2))
        y_out = torch.cat((y_out, out), dim=2)
        return y_out


class Modular2RNN(nn.Module):
    def __init__(self, hparam, in_ch, out_ch, window_size):
        super(Modular2RNN, self).__init__()
        self.out_ch = out_ch

        RNNs = []
        for i in range(out_ch-1):
            RNNs.append(MonolithRNN(hparam=hparam, in_ch=2, out_ch=1, window_size=window_size))
        RNNs.append(MonolithRNN(hparam=hparam, in_ch=7, out_ch=1, window_size=window_size))
        self.RNNs = nn.ModuleList(RNNs)

    def forward(self, x, y):
        for i in range(0, self.out_ch - 1):
            x_in = torch.cat((x[:, :, i].unsqueeze(dim=2), x[:, :, int(i + self.out_ch - 1)].unsqueeze(dim=2)), dim=2)
            out = self.RNNs[i](x_in, y[:, :, i].unsqueeze(dim=2))
            y_out = out if i == 0 else torch.cat((y_out, out), dim=2)

        x_in = torch.cat((x[:, :, self.out_ch - 1].unsqueeze(dim=2), x[:, :, -1].unsqueeze(dim=2), y_out), dim=2)
        out = self.RNNs[self.out_ch - 1](x_in, y[:, :, self.out_ch - 1].unsqueeze(dim=2))
        y_out = torch.cat((y_out, out), dim=2)
        return y_out


class Modular2LSTM(nn.Module):
    def __init__(self, hparam, in_ch, out_ch, window_size):
        super(Modular2LSTM, self).__init__()
        self.out_ch = out_ch

        RNNs = []
        for i in range(out_ch - 1):
            RNNs.append(MonolithLSTM(hparam=hparam, in_ch=2, out_ch=1, window_size=window_size))
        RNNs.append(MonolithLSTM(hparam=hparam, in_ch=7, out_ch=1, window_size=window_size))
        self.RNNs = nn.ModuleList(RNNs)

    def forward(self, x, y):
        for i in range(0, self.out_ch - 1):
            x_in = torch.cat((x[:, :, i].unsqueeze(dim=2), x[:, :, int(i + self.out_ch - 1)].unsqueeze(dim=2)), dim=2)
            out = self.RNNs[i](x_in, y[:, :, i].unsqueeze(dim=2))
            y_out = out if i == 0 else torch.cat((y_out, out), dim=2)

        x_in = torch.cat((x[:, :, self.out_ch - 1].unsqueeze(dim=2), x[:, :, -1].unsqueeze(dim=2), y_out), dim=2)
        out = self.RNNs[self.out_ch - 1](x_in, y[:, :, self.out_ch - 1].unsqueeze(dim=2))
        y_out = torch.cat((y_out, out), dim=2)
        return y_out


class Modular2GRU(nn.Module):
    def __init__(self, hparam, in_ch, out_ch, window_size):
        super(Modular2GRU, self).__init__()
        self.out_ch = out_ch

        RNNs = []
        for i in range(out_ch - 1):
            RNNs.append(MonolithGRU(hparam=hparam, in_ch=2, out_ch=1, window_size=window_size))
        RNNs.append(MonolithGRU(hparam=hparam, in_ch=7, out_ch=1, window_size=window_size))
        self.RNNs = nn.ModuleList(RNNs)

    def forward(self, x, y):
        for i in range(0, self.out_ch - 1):
            x_in = torch.cat((x[:, :, i].unsqueeze(dim=2), x[:, :, int(i + self.out_ch - 1)].unsqueeze(dim=2)), dim=2)
            out = self.RNNs[i](x_in, y[:, :, i].unsqueeze(dim=2))
            y_out = out if i == 0 else torch.cat((y_out, out), dim=2)

        x_in = torch.cat((x[:, :, self.out_ch - 1].unsqueeze(dim=2), x[:, :, -1].unsqueeze(dim=2), y_out), dim=2)
        out = self.RNNs[self.out_ch - 1](x_in, y[:, :, self.out_ch - 1].unsqueeze(dim=2))
        y_out = torch.cat((y_out, out), dim=2)
        return y_out


############################## Modular Type III - Sequential Type #############################################################

class Modular3AE(nn.Module):
    def __init__(self, hparam, in_ch, out_ch, window_size):
        super(Modular3AE, self).__init__()
        self.out_ch = out_ch

        AEs = []
        for i in range(self.out_ch):
            AEs.append(MonolithAE(hparam=hparam, in_ch=2, out_ch=1, window_size=window_size)) if i == 0 \
                else AEs.append(MonolithAE(hparam=hparam, in_ch=3, out_ch=1, window_size=window_size))
        self.AEs = nn.ModuleList(AEs)

    def forward(self, x, y):
        x_in0 = torch.cat((x[:, :, 0].unsqueeze(dim=2), x[:, :, int(self.out_ch - 1)].unsqueeze(dim=2)), dim=2)
        out = self.AEs[0](x_in0, y[:, :, 0].unsqueeze(dim=2))
        y_out = out

        for i in range(1, self.out_ch):
            x_in = torch.cat((x[:, :, i].unsqueeze(dim=2), x[:, :, int(i + self.out_ch - 1)].unsqueeze(dim=2), out), dim=2)
            out = self.AEs[i](x_in, y[:, :, i].unsqueeze(dim=2))
            y_out = torch.cat((y_out, out), dim=2)
        return y_out


class Modular3VAE(nn.Module):
    def __init__(self, hparam, in_ch, out_ch, window_size):
        super(Modular3VAE, self).__init__()
        self.out_ch = out_ch

        AEs = []
        for i in range(self.out_ch):
            AEs.append(MonolithVAE(hparam=hparam, in_ch=2, out_ch=1, window_size=window_size)) if i == 0 \
                else AEs.append(MonolithVAE(hparam=hparam, in_ch=3, out_ch=1, window_size=window_size))
        self.AEs = nn.ModuleList(AEs)

    def forward(self, x, y):
        x_in0 = torch.cat((x[:, :, 0].unsqueeze(dim=2), x[:, :, int(self.out_ch - 1)].unsqueeze(dim=2)), dim=2)
        out = self.AEs[0](x_in0, y[:, :, 0].unsqueeze(dim=2))
        y_out = out

        for i in range(1, self.out_ch):
            x_in = torch.cat((x[:, :, i].unsqueeze(dim=2), x[:, :, int(i + self.out_ch - 1)].unsqueeze(dim=2), out), dim=2)
            out = self.AEs[i](x_in, y[:, :, i].unsqueeze(dim=2))
            y_out = torch.cat((y_out, out), dim=2)
        return y_out


class Modular3RNN(nn.Module):
    def __init__(self, hparam, in_ch, out_ch, window_size):
        super(Modular3RNN, self).__init__()
        self.out_ch = out_ch

        RNNs = []
        for i in range(in_ch):
            RNNs.append(MonolithRNN(hparam=hparam, in_ch=2, out_ch=1, window_size=window_size)) if i == 0 \
                else RNNs.append(MonolithRNN(hparam=hparam, in_ch=3, out_ch=1, window_size=window_size))
        self.RNNs = nn.ModuleList(RNNs)

    def forward(self, x, y):
        x_in0 = torch.cat((x[:, :, 0].unsqueeze(dim=2), x[:, :, int(self.out_ch - 1)].unsqueeze(dim=2)), dim=2)
        out = self.RNNs[0](x_in0, y[:, :, 0].unsqueeze(dim=2))
        y_out = out

        for i in range(1, self.out_ch):
            x_in = torch.cat((x[:, :, i].unsqueeze(dim=2), x[:, :, int(i + self.out_ch - 1)].unsqueeze(dim=2), out), dim=2)
            out = self.RNNs[i](x_in, y[:, :, i].unsqueeze(dim=2))
            y_out = torch.cat((y_out, out), dim=2)
        return y_out


class Modular3LSTM(nn.Module):
    def __init__(self, hparam, in_ch, out_ch, window_size):
        super(Modular3LSTM, self).__init__()
        self.out_ch = out_ch

        RNNs = []
        for i in range(in_ch):
            RNNs.append(MonolithLSTM(hparam=hparam, in_ch=2, out_ch=1, window_size=window_size)) if i == 0 \
                else RNNs.append(MonolithLSTM(hparam=hparam, in_ch=3, out_ch=1, window_size=window_size))
        self.RNNs = nn.ModuleList(RNNs)

    def forward(self, x, y):
        x_in0 = torch.cat((x[:, :, 0].unsqueeze(dim=2), x[:, :, int(self.out_ch - 1)].unsqueeze(dim=2)), dim=2)
        out = self.RNNs[0](x_in0, y[:, :, 0].unsqueeze(dim=2))
        y_out = out

        for i in range(1, self.out_ch):
            x_in = torch.cat((x[:, :, i].unsqueeze(dim=2), x[:, :, int(i + self.out_ch - 1)].unsqueeze(dim=2), out), dim=2)
            out = self.RNNs[i](x_in, y[:, :, i].unsqueeze(dim=2))
            y_out = torch.cat((y_out, out), dim=2)
        return y_out


class Modular3GRU(nn.Module):
    def __init__(self, hparam, in_ch, out_ch, window_size):
        super(Modular3GRU, self).__init__()
        self.out_ch = out_ch

        RNNs = []
        for i in range(in_ch):
            RNNs.append(MonolithGRU(hparam=hparam, in_ch=2, out_ch=1, window_size=window_size)) if i == 0 \
                else RNNs.append(MonolithGRU(hparam=hparam, in_ch=3, out_ch=1, window_size=window_size))
        self.RNNs = nn.ModuleList(RNNs)

    def forward(self, x, y):
        x_in0 = torch.cat((x[:, :, 0].unsqueeze(dim=2), x[:, :, int(self.out_ch - 1)].unsqueeze(dim=2)), dim=2)
        out = self.RNNs[0](x_in0, y[:, :, 0].unsqueeze(dim=2))
        y_out = out

        for i in range(1, self.out_ch):
            x_in = torch.cat((x[:, :, i].unsqueeze(dim=2), x[:, :, int(i + self.out_ch - 1)].unsqueeze(dim=2), out), dim=2)
            out = self.RNNs[i](x_in, y[:, :, i].unsqueeze(dim=2))
            y_out = torch.cat((y_out, out), dim=2)
        return y_out