import React, { useState } from 'react';
import { Upload, X, CheckCircle, Loader2, Image as ImageIcon } from 'lucide-react';

// Define the shape of the data returned by the AI model
interface PredictionResult {
  label: string;
  confidence: number;
  message: string;
  student_info?: {
    reg_number?: string;
    first_name?: string;
    middle_name?: string;
    last_name?: string;
    level?: string;
    error?: string;
  };
}

// Main App Component (Fully compatible with Vite React setup)
const App: React.FC = () => {
  // Explicitly type the state variables
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  /**
   * Handles file selection, setting the file state and creating a temporary
   * URL for image preview.
   * @param {React.ChangeEvent<HTMLInputElement>} event
   */
  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    // We check for event.target.files before accessing [0]
    const selectedFile = event.target.files ? event.target.files[0] : null;

    // Reset states on new file selection
    setResult(null);
    setError(null);
    setFile(selectedFile);

    if (previewUrl) {
      URL.revokeObjectURL(previewUrl); // Clean up previous preview URL
    }

    if (selectedFile) {
      setPreviewUrl(URL.createObjectURL(selectedFile));
    } else {
      setPreviewUrl(null);
    }
  };

  /**
   * Simulates the API call to the backend for AI prediction.
   */
  const handleUpload = async () => {
    if (!file) {
      setError("Please select an image file to analyze.");
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const apiResponse = await fetch('https://dubem.getmusterup.com/predict', {
        method: 'POST',
        body: formData,
      });

      if (!apiResponse.ok) {
        const errorData = await apiResponse.json();
        throw new Error(errorData.detail || `Server responded with status: ${apiResponse.status}`);
      }

      const data: PredictionResult = await apiResponse.json();
      setResult(data);

    } catch (err: unknown) {
      console.error("Upload Error:", err);
      if (err instanceof Error) {
        setError(`Failed to connect to the AI model server: ${err.message}`);
      } else {
        setError("An unknown error occurred during upload.");
      }
    } finally {
      setLoading(false);
    }
  };

  /**
   * Clears the selected file and preview.
   */
  const handleClear = () => {
    setFile(null);
    setPreviewUrl(null);
    setResult(null);
    setError(null);
    // Explicitly cast to HTMLInputElement to access .value property
    const fileInput = document.getElementById('file-input') as HTMLInputElement | null;
    if (fileInput) {
        fileInput.value = ''; // Clear input element
    }
  };

  // Convert confidence (0.0 to 1.0) string to percentage string
  const getConfidenceText = (conf: string): string => {
    const percentage = (parseFloat(conf) * 100).toFixed(2);
    return `${percentage}%`;
  };

  return (
    <div className="min-h-screen bg-gray-100 text-gray-800 font-sans">
      <main className="container mx-auto p-4 md:p-8">
        <div className="max-w-4xl mx-auto">
          <header className="text-center mb-8">
            <h1 className="text-4xl md:text-5xl font-bold text-gray-900">Handwriting Verification</h1>
            <p className="text-lg text-gray-600 mt-2">Upload a document to verify the author's handwriting against our records.</p>
          </header>

          <div className="bg-white rounded-2xl shadow-xl p-6 md:p-8 border border-gray-200">
            <div className="grid md:grid-cols-2 gap-8">
              {/* Left Side: Uploader and Preview */}
              <div className="flex flex-col">
                <div
                  className="relative border-2 border-dashed border-gray-300 rounded-xl p-6 text-center cursor-pointer hover:border-blue-500 hover:bg-blue-50 transition-all duration-300"
                  onClick={() => document.getElementById('file-input')?.click()}
                >
                  <input
                    id="file-input"
                    type="file"
                    accept="image/*"
                    onChange={handleFileChange}
                    className="hidden"
                  />
                  <div className="flex flex-col items-center justify-center h-full">
                    <Upload className="w-12 h-12 text-gray-400 mb-4" />
                    <p className="font-semibold text-gray-700">
                      {file ? file.name : 'Click to upload or drag and drop'}
                    </p>
                    <p className="text-sm text-gray-500">PNG, JPG, WEBP (max. 10MB)</p>
                  </div>
                </div>

                {previewUrl && (
                  <div className="mt-6">
                    <h3 className="font-semibold text-lg mb-2">Image Preview</h3>
                    <div className="bg-gray-100 rounded-lg p-2 border">
                      <img
                        src={previewUrl}
                        alt="Selected Preview"
                        className="w-full h-auto max-h-64 object-contain rounded"
                      />
                    </div>
                  </div>
                )}

                <div className="flex gap-4 mt-6">
                  <button
                    onClick={handleUpload}
                    disabled={!file || loading}
                    className="flex-1 bg-blue-600 text-white font-bold py-3 px-4 rounded-lg hover:bg-blue-700 disabled:bg-blue-300 disabled:cursor-not-allowed transition-all duration-300 flex items-center justify-center gap-2 shadow-md"
                  >
                    {loading ? (
                      <>
                        <Loader2 className="animate-spin" />
                        <span>Verifying...</span>
                      </>
                    ) : (
                      'Verify Handwriting'
                    )}
                  </button>
                  {file && (
                    <button
                      onClick={handleClear}
                      className="p-3 bg-gray-200 text-gray-700 rounded-lg hover:bg-gray-300 transition-colors"
                    >
                      <X />
                    </button>
                  )}
                </div>
              </div>

              {/* Right Side: Results */}
              <div className="bg-gray-50 rounded-xl p-6 border">
                <h2 className="text-2xl font-bold text-gray-800 mb-4">Verification Result</h2>
                <div className="h-full flex flex-col justify-center">
                  {error && (
                    <div className="text-center p-4 bg-red-100 text-red-700 rounded-lg">
                      <p className="font-bold">Error</p>
                      <p>{error}</p>
                    </div>
                  )}
                  {loading && (
                    <div className="text-center text-gray-500">
                      <Loader2 className="w-8 h-8 animate-spin mx-auto mb-2" />
                      <p>Analyzing image...</p>
                    </div>
                  )}
                  {!loading && !error && !result && (
                    <div className="text-center text-gray-500">
                      <ImageIcon className="w-12 h-12 mx-auto mb-2" />
                      <p>Results will be displayed here.</p>
                    </div>
                  )}
                  {result && (
                    <div className="space-y-6">
                      <div>
                        <p className="text-sm text-gray-500">Verdict</p>
                        <p className={`text-2xl font-bold ${result.label === 'Unrecognized' ? 'text-red-600' : 'text-green-600'}`}>
                          {result.label}
                        </p>
                        <p className="text-sm text-gray-600 mt-1">{result.message}</p>
                      </div>

                      {result.student_info && !result.student_info.error && (
                        <div>
                          <h3 className="text-lg font-bold text-gray-800 border-b pb-2 mb-3">Student Information</h3>
                          <div className="space-y-2 text-sm">
                            <div className="flex justify-between">
                              <span className="font-semibold text-gray-600">Full Name:</span>
                              <span className="text-gray-900">{`${result.student_info.first_name || ''} ${result.student_info.middle_name || ''} ${result.student_info.last_name || ''}`}</span>
                            </div>
                            <div className="flex justify-between">
                              <span className="font-semibold text-gray-600">Reg. Number:</span>
                              <span className="text-gray-900">{result.student_info.reg_number}</span>
                            </div>
                            <div className="flex justify-between">
                              <span className="font-semibold text-gray-600">Level:</span>
                              <span className="text-gray-900">{result.student_info.level}</span>
                            </div>
                          </div>
                        </div>
                      )}
                      {result.student_info?.error && (
                         <div className="text-center p-4 bg-yellow-100 text-yellow-700 rounded-lg">
                           <p>{result.student_info.error}</p>
                         </div>
                      )}
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
};

export default App;
