import torch

class DataLoader():
    def __init__(self, data_set, shuffle, batch_size):
        self.shuffle = shuffle
        self.batch_size = batch_size

        self.AABB_lower = data_set.AABB_lower
        self.AABB_upper = data_set.AABB_upper
        self.points = data_set.points
        self.radii = data_set.radii

     
    def get_num_points(self):
        return len(self.points)
    
    def adapt_number_of_batches_to(self, other_data_loader):
        n_batches_other = len(other_data_loader)
        n_points_self = self.get_num_points()
        
        # Find new batch size for self that results in the same number of batches as other_data_loader
        new_batch_size_self = n_points_self // n_batches_other
        
        # Adjust new_batch_size_self to get as close as possible to n_batches_other while using all points
        if self.get_num_points() > other_data_loader.get_num_points():
            while n_batches_other < n_points_self/new_batch_size_self:
                new_batch_size_self +=1   
        else:
            while n_batches_other > n_points_self/new_batch_size_self:
                new_batch_size_self -=1
            new_batch_size_self +=1

        self.batch_size = new_batch_size_self

    def __len__(self):
        return len(self.points) // self.batch_size + (1 if len(self.points) % self.batch_size != 0 else 0)
    
    def __getitem__(self, idx):
        if self.radii is None:
            return self.points[idx], None
        return self.points[idx], self.radii[idx]
    
    def __iter__(self):
        if self.shuffle:
            self._shuffle_data()

        # full batches
        for i in range(len(self)):
            start = i * self.batch_size
            end = (i + 1) * self.batch_size
            if self.radii is not None:
                yield self.points[start:end, :], self.radii[start:end]
            else:
                yield self.points[start:end, :], None

        # last batch
        if len(self) * self.batch_size < len(self.points):
            start = len(self) * self.batch_size
            end = len(self.points)
            if not self.random and self.points_inside_tet_mesh:
                yield self.points[start:end, :], self.radii[start:end]
            yield self.points[start:end, :], None

    def _shuffle_data(self):
        number_of_rows = self.points.size()[0]
        random_permutation_of_indices = torch.randperm(number_of_rows)
        self.points = self.points[random_permutation_of_indices]
        if not self.radii is None:
            self.radii = self.radii[random_permutation_of_indices]    