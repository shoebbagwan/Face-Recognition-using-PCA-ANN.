import numpy as np
import cv2
import os
from scipy.linalg import eigh
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import glob

class PCA_ANN_FaceRecognition:
    def __init__(self):
        self.mean_face = None
        self.eigenfaces = None
        self.feature_vector = None
        self.signatures = None
        self.labels = None
        self.ann_model = None
        self.img_height = None
        self.img_width = None
        
    def load_dataset(self, dataset_path):
        """
        Load face images from the dataset directory
        
        Parameters:
        -----------
        dataset_path : str
            Path to the dataset folder containing subdirectories for each person
            Example structure: dataset/person1/*.jpg, dataset/person2/*.jpg, etc.
        
        Returns:
        --------
        face_db : numpy array of shape (m*n, p)
        labels : list of labels for each image
        """
        print("Loading dataset...")
        faces = []
        labels = []
        label_names = []
        
        # Get all subdirectories (each represents a person)
        person_dirs = [d for d in os.listdir(dataset_path) 
                      if os.path.isdir(os.path.join(dataset_path, d))]
        person_dirs.sort()
        
        for idx, person_dir in enumerate(person_dirs):
            person_path = os.path.join(dataset_path, person_dir)
            image_files = glob.glob(os.path.join(person_path, '*.*'))
            
            for img_file in image_files:
                # Read image in grayscale
                img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    # Resize to standard size (you can adjust this)
                    img = cv2.resize(img, (92, 112))
                    
                    # Store dimensions
                    if self.img_height is None:
                        self.img_height, self.img_width = img.shape
                    
                    # Flatten image to column vector
                    img_vector = img.flatten().reshape(-1, 1)
                    faces.append(img_vector)
                    labels.append(idx)
            
            label_names.append(person_dir)
        
        # Convert to numpy array: shape (m*n, p)
        face_db = np.hstack(faces)
        labels = np.array(labels)
        
        print(f"Loaded {face_db.shape[1]} images from {len(person_dirs)} persons")
        print(f"Image dimensions: {self.img_height} x {self.img_width}")
        print(f"Face database shape: {face_db.shape}")
        
        return face_db, labels, label_names
    
    def calculate_mean_face(self, face_db):
        """
        Calculate mean face from the face database
        
        Parameters:
        -----------
        face_db : numpy array of shape (m*n, p)
        
        Returns:
        --------
        mean_face : numpy array of shape (m*n, 1)
        """
        print("\nCalculating mean face...")
        mean_face = np.mean(face_db, axis=1, keepdims=True)
        return mean_face
    
    def mean_zero_alignment(self, face_db, mean_face):
        """
        Subtract mean face from each face image
        
        Parameters:
        -----------
        face_db : numpy array of shape (m*n, p)
        mean_face : numpy array of shape (m*n, 1)
        
        Returns:
        --------
        phi : numpy array of shape (m*n, p)
        """
        print("Performing mean zero alignment...")
        phi = face_db - mean_face
        return phi
    
    def calculate_covariance(self, phi):
        """
        Calculate surrogate covariance matrix
        
        Parameters:
        -----------
        phi : numpy array of shape (m*n, p)
        
        Returns:
        --------
        C : numpy array of shape (p, p)
        """
        print("Calculating covariance matrix...")
        # Surrogate covariance: C = phi^T * phi
        C = np.dot(phi.T, phi)
        return C
    
    def eigen_decomposition(self, C):
        """
        Perform eigenvalue and eigenvector decomposition
        
        Parameters:
        -----------
        C : numpy array of shape (p, p)
        
        Returns:
        --------
        eigenvalues : numpy array of shape (p,)
        eigenvectors : numpy array of shape (p, p)
        """
        print("Performing eigen decomposition...")
        eigenvalues, eigenvectors = eigh(C)
        
        # Sort in descending order
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        return eigenvalues, eigenvectors
    
    def select_best_directions(self, eigenvalues, eigenvectors, k):
        """
        Select k best eigenvectors based on eigenvalues
        
        Parameters:
        -----------
        eigenvalues : numpy array of shape (p,)
        eigenvectors : numpy array of shape (p, p)
        k : int, number of components to keep
        
        Returns:
        --------
        feature_vector : numpy array of shape (p, k)
        """
        print(f"Selecting top {k} eigenvectors...")
        feature_vector = eigenvectors[:, :k]
        
        # Calculate explained variance
        total_variance = np.sum(eigenvalues)
        explained_variance = np.sum(eigenvalues[:k]) / total_variance * 100
        print(f"Explained variance with {k} components: {explained_variance:.2f}%")
        
        return feature_vector
    
    def generate_eigenfaces(self, phi, feature_vector):
        """
        Generate eigenfaces by projecting mean-aligned faces to feature vector
        
        Parameters:
        -----------
        phi : numpy array of shape (m*n, p)
        feature_vector : numpy array of shape (p, k)
        
        Returns:
        --------
        eigenfaces : numpy array of shape (k, m*n)
        """
        print("Generating eigenfaces...")
        eigenfaces = np.dot(feature_vector.T, phi.T)
        return eigenfaces
    
    def generate_signatures(self, eigenfaces, phi):
        """
        Generate signature for each face
        
        Parameters:
        -----------
        eigenfaces : numpy array of shape (k, m*n)
        phi : numpy array of shape (m*n, p)
        
        Returns:
        --------
        signatures : numpy array of shape (k, p)
        """
        print("Generating signatures...")
        signatures = np.dot(eigenfaces, phi)
        return signatures
    
    def train(self, dataset_path, k=50, hidden_layers=(100, 50), max_iter=500):
        """
        Train the PCA-ANN face recognition system
        
        Parameters:
        -----------
        dataset_path : str
            Path to dataset directory
        k : int
            Number of principal components to keep
        hidden_layers : tuple
            Architecture of hidden layers in ANN
        max_iter : int
            Maximum iterations for ANN training
        """
        # Step 1: Load dataset
        face_db, self.labels, self.label_names = self.load_dataset(dataset_path)
        
        # Step 2: Calculate mean face
        self.mean_face = self.calculate_mean_face(face_db)
        
        # Step 3: Mean zero alignment
        phi = self.mean_zero_alignment(face_db, self.mean_face)
        
        # Step 4: Calculate covariance
        C = self.calculate_covariance(phi)
        
        # Step 5: Eigen decomposition
        eigenvalues, eigenvectors = self.eigen_decomposition(C)
        
        # Step 6: Select best directions
        self.feature_vector = self.select_best_directions(eigenvalues, eigenvectors, k)
        
        # Step 7: Generate eigenfaces
        self.eigenfaces = self.generate_eigenfaces(phi, self.feature_vector)
        
        # Step 8: Generate signatures
        self.signatures = self.generate_signatures(self.eigenfaces, phi)
        
        # Step 9: Train ANN
        print("\nTraining ANN model...")
        X_train = self.signatures.T  # Shape: (p, k)
        y_train = self.labels
        
        self.ann_model = MLPClassifier(
            hidden_layer_sizes=hidden_layers,
            max_iter=max_iter,
            random_state=42,
            verbose=False,
            early_stopping=True,
            validation_fraction=0.1
        )
        
        self.ann_model.fit(X_train, y_train)
        
        train_accuracy = accuracy_score(y_train, self.ann_model.predict(X_train))
        print(f"Training accuracy: {train_accuracy*100:.2f}%")
        
    def predict(self, test_image_path):
        """
        Predict the identity of a test face image
        
        Parameters:
        -----------
        test_image_path : str
            Path to test image
        
        Returns:
        --------
        predicted_label : int
        confidence : float
        """
        # Step 1: Load test image
        img = cv2.imread(test_image_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (self.img_width, self.img_height))
        img_vector = img.flatten().reshape(-1, 1)
        
        # Step 2: Mean zero alignment
        img_aligned = img_vector - self.mean_face
        
        # Step 3: Project to eigenfaces
        projected = np.dot(self.eigenfaces, img_aligned)
        
        # Step 4: Predict using ANN
        prediction = self.ann_model.predict(projected.T)
        probabilities = self.ann_model.predict_proba(projected.T)
        confidence = np.max(probabilities)
        
        return prediction[0], confidence
    
    def evaluate_with_train_test_split(self, dataset_path, k=50, test_size=0.4):
        """
        Evaluate the model with 60-40 train-test split
        
        Parameters:
        -----------
        dataset_path : str
            Path to dataset directory
        k : int
            Number of principal components
        test_size : float
            Proportion of test set (0.4 = 40%)
        
        Returns:
        --------
        accuracy : float
        """
        print(f"\n{'='*60}")
        print(f"Evaluating with k={k} principal components")
        print(f"{'='*60}")
        
        # Load full dataset
        face_db, labels, label_names = self.load_dataset(dataset_path)
        
        # Split dataset
        indices = np.arange(face_db.shape[1])
        train_idx, test_idx = train_test_split(
            indices, test_size=test_size, random_state=42, stratify=labels
        )
        
        train_faces = face_db[:, train_idx]
        train_labels = labels[train_idx]
        test_faces = face_db[:, test_idx]
        test_labels = labels[test_idx]
        
        print(f"\nTraining set: {len(train_idx)} images")
        print(f"Test set: {len(test_idx)} images")
        
        # Training phase
        self.mean_face = self.calculate_mean_face(train_faces)
        phi_train = self.mean_zero_alignment(train_faces, self.mean_face)
        C = self.calculate_covariance(phi_train)
        eigenvalues, eigenvectors = self.eigen_decomposition(C)
        self.feature_vector = self.select_best_directions(eigenvalues, eigenvectors, k)
        self.eigenfaces = self.generate_eigenfaces(phi_train, self.feature_vector)
        signatures_train = self.generate_signatures(self.eigenfaces, phi_train)
        
        # Train ANN
        print("\nTraining ANN...")
        self.ann_model = MLPClassifier(
            hidden_layer_sizes=(100, 50),
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1
        )
        self.ann_model.fit(signatures_train.T, train_labels)
        
        # Testing phase
        print("\nTesting...")
        phi_test = self.mean_zero_alignment(test_faces, self.mean_face)
        signatures_test = self.generate_signatures(self.eigenfaces, phi_test)
        
        predictions = self.ann_model.predict(signatures_test.T)
        accuracy = accuracy_score(test_labels, predictions)
        
        print(f"\nTest Accuracy: {accuracy*100:.2f}%")
        return accuracy


def evaluate_different_k_values(dataset_path, k_values):
    """
    Evaluate model with different k values and plot results
    
    Parameters:
    -----------
    dataset_path : str
        Path to dataset directory
    k_values : list
        List of k values to evaluate
    """
    accuracies = []
    
    for k in k_values:
        model = PCA_ANN_FaceRecognition()
        accuracy = model.evaluate_with_train_test_split(dataset_path, k=k)
        accuracies.append(accuracy)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, [acc*100 for acc in accuracies], marker='o', linewidth=2, markersize=8)
    plt.xlabel('Number of Principal Components (k)', fontsize=12)
    plt.ylabel('Classification Accuracy (%)', fontsize=12)
    plt.title('PCA-ANN Face Recognition: Accuracy vs K Value', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('accuracy_vs_k.png', dpi=300)
    plt.show()
    
    print("\n" + "="*60)
    print("Summary of Results:")
    print("="*60)
    for k, acc in zip(k_values, accuracies):
        print(f"k = {k:3d}  -->  Accuracy = {acc*100:.2f}%")


def main():
    # =================================================================
    # DATASET PATH - MODIFY THIS TO POINT TO YOUR DATASET LOCATION
    # =================================================================
    dataset_path = "path/to/your/dataset"  # <-- CHANGE THIS PATH
    
    # Example: dataset_path = "C:/Users/YourName/Desktop/dataset"
    # or: dataset_path = "/home/user/datasets/face_dataset"
    
    print("PCA-ANN Face Recognition System")
    print("="*60)
    
    # Part (a): Evaluate with different k values
    print("\nPart (a): Evaluating with different k values...")
    k_values = [10, 20, 30, 40, 50, 75, 100, 150]
    evaluate_different_k_values(dataset_path, k_values)
    
    # Part (b): Train final model and test with imposters
    print("\n" + "="*60)
    print("Part (b): Training final model...")
    print("="*60)
    
    model = PCA_ANN_FaceRecognition()
    model.train(dataset_path, k=50)
    
    # =================================================================
    # IMPOSTER DETECTION
    # To test with imposters (people not in training set):
    # 1. Place imposter images in a separate folder
    # 2. Uncomment and modify the code below:
    # =================================================================
    
    # imposter_folder = "path/to/imposter/images"  # <-- CHANGE THIS
    # imposter_images = glob.glob(os.path.join(imposter_folder, '*.*'))
    # 
    # print("\nTesting with imposters...")
    # threshold = 0.5  # Confidence threshold for recognition
    # 
    # for img_path in imposter_images:
    #     predicted_label, confidence = model.predict(img_path)
    #     
    #     if confidence < threshold:
    #         print(f"{os.path.basename(img_path)}: NOT ENROLLED (confidence: {confidence:.2f})")
    #     else:
    #         print(f"{os.path.basename(img_path)}: Recognized as {model.label_names[predicted_label]} (confidence: {confidence:.2f})")
    
    print("\n" + "="*60)
    print("Training and evaluation completed!")
    print("="*60)


if __name__ == "__main__":
    main()