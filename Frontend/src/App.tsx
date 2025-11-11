import React, { useState } from 'react';
import { Upload, X, Loader2 } from 'lucide-react';

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
      const apiResponse = await fetch('/predict', {
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

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100 font-sans flex items-center justify-center p-4 sm:p-6 lg:p-8">
      <div className="w-full max-w-lg">
        <header className="text-center mb-8">
          <h1 className="text-4xl font-extrabold text-gray-900 leading-tight">Handwriting <span className="text-blue-600">Verifier</span></h1>
          <p className="text-lg text-gray-600 mt-2">Securely authenticate documents by analyzing unique handwriting patterns.</p>
        </header>

        <div className="bg-white border border-gray-200 rounded-3xl shadow-xl p-6 sm:p-8 lg:p-10 text-center relative overflow-hidden">
          {/* Decorative background element */}
          <div className="absolute top-0 left-0 w-full h-full bg-gradient-to-br from-blue-50 to-transparent rounded-3xl opacity-50"></div>

          <div className="relative z-10">
            {/* Uploader Section */}
            {!previewUrl && (
              <div
                className="border-2 border-dashed border-blue-300 rounded-2xl p-10 cursor-pointer hover:border-blue-500 bg-blue-50 transition-all duration-300 ease-in-out group"
                onClick={() => document.getElementById('file-input')?.click()}
              >
                <input id="file-input" type="file" accept="image/*" onChange={handleFileChange} className="hidden" />
                <div className="flex flex-col items-center text-blue-600 group-hover:text-blue-800 transition-colors duration-300">
                  <Upload className="w-12 h-12 mb-4 group-hover:scale-110 transition-transform duration-300" />
                  <p className="font-semibold text-lg">Click to upload</p>
                  <p className="text-sm text-gray-500 mt-1">PNG, JPG, WEBP (max. 10MB)</p>
                </div>
              </div>
            )}

            {/* Preview & Actions Section */}
            {previewUrl && (
              <div className="space-y-6">
                <div className="bg-gray-100 rounded-xl p-3 border border-gray-200 w-full aspect-video flex items-center justify-center overflow-hidden shadow-inner">
                  <img src={previewUrl} alt="Document Preview" className="max-w-full max-h-full object-contain rounded-lg" />
                </div>
                <div className="flex gap-4">
                  <button
                    onClick={handleUpload}
                    disabled={loading}
                    className="flex-1 bg-blue-600 text-white font-bold py-3 px-6 rounded-xl hover:bg-blue-700 disabled:bg-blue-300 transition-all duration-300 ease-in-out flex items-center justify-center gap-2 shadow-md hover:shadow-lg"
                  >
                    {loading ? <Loader2 className="animate-spin w-5 h-5" /> : <span className="text-lg">Verify Document</span>}
                  </button>
                  <button
                    onClick={handleClear}
                    className="p-3 bg-gray-200 text-gray-700 rounded-xl hover:bg-gray-300 transition-colors duration-300 shadow-sm"
                  >
                    <X className="w-6 h-6" />
                  </button>
                </div>
              </div>
            )}

            {/* Loading State */}
            {loading && (
              <div className="absolute inset-0 bg-white bg-opacity-80 flex flex-col items-center justify-center rounded-3xl z-20 animate-fade-in">
                <Loader2 className="w-16 h-16 text-blue-500 animate-spin mb-4" />
                <p className="text-xl font-semibold text-gray-700">Analyzing Handwriting...</p>
                <p className="text-sm text-gray-500 mt-2">This may take a few moments.</p>
              </div>
            )}

            {/* Results Panel */}
            {result && !loading && (
              <div className="mt-8 text-left animate-fade-in-up">
                <h2 className="text-2xl font-bold text-gray-800 mb-4 border-b pb-2">Verification Outcome</h2>
                <div className="space-y-5">
                  {/* Verdict */}
                  <div className={`p-4 rounded-xl ${result.label === 'Unrecognized' ? 'bg-red-50 text-red-800 border border-red-200' : 'bg-green-50 text-green-800 border border-green-200'}`}>
                    <p className="font-bold text-xl mb-1">{result.label}</p>
                    <p className="text-sm">{result.message}</p>
                  </div>

                  {/* Student Info (if available) */}
                  {result.student_info && !result.student_info.error && (
                    <div className="bg-gray-50 p-5 rounded-xl border border-gray-200 shadow-sm">
                      <h3 className="font-semibold text-gray-700 mb-3 text-lg">Student Details</h3>
                      <div className="text-sm space-y-2">
                        <div className="flex justify-between items-center">
                          <span className="text-gray-500">Full Name:</span>
                          <span className="font-medium text-gray-800">{`${result.student_info.first_name || ''} ${result.student_info.middle_name || ''} ${result.student_info.last_name || ''}`}</span>
                        </div>
                        <div className="flex justify-between items-center">
                          <span className="text-gray-500">Reg. Number:</span>
                          <span className="font-medium text-gray-800">{result.student_info.reg_number}</span>
                        </div>
                        <div className="flex justify-between items-center">
                          <span className="text-gray-500">Level:</span>
                          <span className="font-medium text-gray-800">{result.student_info.level}</span>
                        </div>
                      </div>
                    </div>
                  )}
                  {result.student_info?.error && (
                    <div className="p-4 bg-yellow-50 text-yellow-800 rounded-xl border border-yellow-200 text-sm">
                      <p className="font-semibold">Information Unavailable:</p>
                      <p>{result.student_info.error}</p>
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* General Error Display */}
            {error && !loading && (
              <div className="mt-8 text-left animate-fade-in-up">
                <div className="p-4 bg-red-50 text-red-800 rounded-xl border border-red-200">
                  <p className="font-bold text-xl mb-1">Error Occurred</p>
                  <p className="text-sm">{error}</p>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default App;
