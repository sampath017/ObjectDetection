import torch
import pandas as pd


class KnnClassifier:
    def __init__(self, metrics_file):
        self.metrics_file = metrics_file
        self.metrics_df = pd.read_csv(metrics_file)

    def train(self, train_dataloader):
        self.train_dataloader = train_dataloader

    def predict(self, val_dataloader, k):
        if not any(self.metrics_df["k"] == k):
            accuracies = []
            val_y_s = []
            val_pred_s = []
            for batch, (val_x, val_y) in enumerate(val_dataloader):
                val_y_s.extend(val_y)
                num_val = val_x.shape[0]
                A = val_x.reshape(num_val, -1)

                dists_list = []
                train_y_list = []
                for train_x, train_y in self.train_dataloader:
                    train_y_list.append(train_y)
                    num_train = train_x.shape[0]

                    B = train_x.reshape(num_train, -1)

                    AB2 = A.mm(B.T)*2

                    # dists[i, j] is the distance between ith val point and jth train point
                    dists = ((A**2).sum(dim=1).reshape(-1, 1) - AB2 +
                             # (val, train)
                             (B**2).sum(dim=1).reshape(1, -1))**(1/2)

                    dists_list.append(dists)

                train_y = torch.cat(train_y_list, dim=0)
                dists = torch.cat(dists_list, dim=-1)

                indices = dists.topk(k=k, dim=-1, largest=False).indices

                val_pred = torch.empty(num_val)
                for i in range(indices.shape[0]):
                    _, val_pred[i] = torch.max(
                        train_y[indices[i]].bincount(), dim=0)

                val_pred_s.extend(val_pred)
                accuracy = 100 * (sum(val_pred == val_y) / val_y.shape[0])
                accuracies.append(accuracy)
                print(f"Batch {batch}: {accuracy:.2f}")

            final_accuracy = torch.mean(torch.tensor(accuracies)).item()
            print(f"Accuracy on full dataset: {final_accuracy:.2f}")

            # Append k and accuracy to metrics dataframe
            new_metrics = pd.DataFrame(
                {'k': [k], 'accuracy': [final_accuracy]})
            self.metrics_df = pd.concat(
                [self.metrics_df, new_metrics], ignore_index=True)

            # Save updated metrics to the file
            self.metrics_df.to_csv(self.metrics_file, index=False)

            return final_accuracy, val_y_s, val_pred_s
        else:
            print(f"For k={k} the accuracy already measured!.")
