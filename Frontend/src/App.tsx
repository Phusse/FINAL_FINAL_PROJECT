import React, { useState } from 'react';
import { Upload, X, Loader2, Image as ImageIcon } from 'lucide-react';

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

  return (
    <div className="min-h-screen bg-[#F7F7F8] font-sans flex items-center justify-center p-4">
      <div className="w-full max-w-md">
        <div className="bg-white border border-gray-200 rounded-2xl shadow-lg text-center p-8">
          <h1 className="text-2xl font-bold text-gray-800 mb-2">Handwriting Verification</h1>
          <p className="text-gray-500 mb-6">Upload an image to verify the writer's identity.</p>

          {/* Uploader */}
          {!previewUrl && (
            <div
              className="relative border-2 border-dashed border-gray-300 rounded-xl p-10 cursor-pointer hover:border-blue-500 bg-gray-50 transition-all"
              onClick={() => document.getElementById('file-input')?.click()}
            >
              <input id="file-input" type="file" accept="image/*" onChange={handleFileChange} className="hidden" />
              <div className="flex flex-col items-center text-gray-500">
                <Upload className="w-10 h-10 mb-3" />
                <p className="font-semibold">Click to upload</p>
                <p className="text-sm">PNG, JPG, WEBP</p>
              </div>
            </div>
          )}

          {/* Preview & Actions */}
          {previewUrl && (
            <div className="space-y-4">
              <div className="bg-gray-100 rounded-lg p-2 border w-full aspect-video overflow-hidden">
                <img src={previewUrl} alt="Preview" className="w-full h-full object-contain rounded" />
              </div>
              <div className="flex gap-3">
                <button
                  onClick={handleUpload}
                  disabled={loading}
                  className="flex-1 bg-blue-600 text-white font-bold py-3 px-4 rounded-lg hover:bg-blue-700 disabled:bg-blue-400 transition-all flex items-center justify-center gap-2"
                >
                  {loading ? <Loader2 className="animate-spin" /> : 'Verify'}
                </button>
                <button onClick={handleClear} className="p-3 bg-gray-200 text-gray-600 rounded-lg hover:bg-gray-300 transition-colors">
                  <X />
                </button>
              </div>
            </div>
          )}
        </div>

        {/* Results Panel */}
        {result && !loading && (
          <div className="bg-white border border-gray-200 rounded-2xl shadow-lg p-8 mt-6 animate-fade-in">
            <h2 className="text-xl font-bold text-gray-800 mb-4">Result</h2>
            <div className="space-y-4">
              {/* Verdict */}
              <div className={`p-4 rounded-lg ${result.label === 'Unrecognized' ? 'bg-red-100 text-red-800' : 'bg-green-100 text-green-800'}`}>
                <p className="font-bold text-lg">{result.label}</p>
                <p className="text-sm">{result.message}</p>
              </div>

              {/* Student Info */}
              {result.student_info && !result.student_info.error && (
                <div>
                  <h3 className="font-semibold text-gray-700 mb-2">Student Details</h3>
                  <div className="text-sm space-y-2 text-left bg-gray-50 p-4 rounded-lg border">
                    <div className="flex justify-between">
                      <span className="text-gray-500">Full Name:</span>
                      <span className="font-medium text-gray-800">{`${result.student_info.first_name || ''} ${result.student_info.middle_name || ''} ${result.student_info.last_name || ''}`}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-500">Reg. Number:</span>
                      <span className="font-medium text-gray-800">{result.student_info.reg_number}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-500">Level:</span>
                      <span className="font-medium text-gray-800">{result.student_info.level}</span>
                    </div>
                  </div>
                </div>
              )}
               {result.student_info?.error && (
                  <div className="p-3 bg-yellow-100 text-yellow-800 rounded-lg text-sm">
                    <p>{result.student_info.error}</p>
                  </div>
              )}
            </div>
          </div>
        )}
         {error && !loading && (
            <div className="bg-red-100 border border-red-200 text-red-800 rounded-2xl shadow-lg p-8 mt-6 animate-fade-in">
                 <p className="font-bold">Error</p>
                 <p>{error}</p>
            </div>
         )}
      </div>
    </div>
  );
};

export default App;
