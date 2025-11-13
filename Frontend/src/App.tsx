import React, { useState } from 'react';
import {
  Upload,
  X,
  Loader2,
  XCircle,
  FileCheck,
  CheckCircle,
} from 'lucide-react';

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
    passport_url?: string;
    error?: string;
  };
}

// Main App Component (Fully compatible with Vite React setup)
const App: React.FC = () => {
  // === ALL YOUR STATE AND LOGIC IS UNCHANGED ===
  // (It's already well-written)
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = event.target.files ? event.target.files[0] : null;

    setResult(null);
    setError(null);
    setFile(selectedFile);

    if (previewUrl) {
      URL.revokeObjectURL(previewUrl);
    }

    if (selectedFile) {
      setPreviewUrl(URL.createObjectURL(selectedFile));
    } else {
      setPreviewUrl(null);
    }
  };

  const handleUpload = async () => {
    if (!file) {
      setError('Please select an image file to analyze.');
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const apiResponse = await fetch('https://api.eceexams.online/predict', {
        method: 'POST',
        body: formData,
      });

      if (!apiResponse.ok) {
        const errorData = await apiResponse.json();
        throw new Error(
          errorData.detail ||
            `Server responded with status: ${apiResponse.status}`
        );
      }

      const data: PredictionResult = await apiResponse.json();
      setResult(data);
    } catch (err: unknown) {
      console.error('Upload Error:', err);
      if (err instanceof Error) {
        setError(`Failed to connect to the AI model server: ${err.message}`);
      } else {
        setError('An unknown error occurred during upload.');
      }
    } finally {
      setLoading(false);
    }
  };

  const handleClear = () => {
    setFile(null);
    setPreviewUrl(null);
    setResult(null);
    setError(null);
    const fileInput = document.getElementById(
      'file-input'
    ) as HTMLInputElement | null;
    if (fileInput) {
      fileInput.value = '';
    }
  };

  // === NEW JSX / VISUALS START HERE ===

  /**
   * This helper function conditionally renders the "Right Panel"
   * based on the app's state (loading, error, result, or initial).
   */
  const renderResultPanel = () => {
    // 1. Loading State
    if (loading) {
      return (
        <div className="flex flex-col items-center justify-center h-full gap-4 text-center">
          <Loader2 className="w-16 h-16 text-emerald-600 animate-spin" />
          <h3 className="text-2xl font-bold text-gray-800">
            Analyzing Handwriting...
          </h3>
          <p className="text-gray-500">
            Our AI is verifying the document. This may take a moment.
          </p>
        </div>
      );
    }

    // 2. Error State
    if (error) {
      return (
        <div className="p-6 bg-red-50 border border-red-200 rounded-2xl h-full flex flex-col justify-center text-center">
          <div className="w-16 h-16 bg-red-100 text-red-600 rounded-full flex items-center justify-center mx-auto mb-4">
            <XCircle className="w-8 h-8" />
          </div>
          <h3 className="text-xl font-bold text-red-800 mb-2">
            Error Occurred
          </h3>
          <p className="text-red-700 text-sm">{error}</p>
        </div>
      );
    }

    // 3. Result State
    if (result) {
      const isVerified = result.label !== 'Unrecognized';

      return (
        <div className="animate-fade-in-up">
          <h2 className="text-3xl font-bold text-gray-900 mb-6">
            Verification Outcome
          </h2>
          <div
            className={`shadow-lg rounded-2xl overflow-hidden border ${
              isVerified
                ? 'border-green-300'
                : 'border-red-300'
            }`}
          >
            {/* Verdict Header */}
            <div
              className={`p-5 flex items-center gap-4 ${
                isVerified
                  ? 'bg-green-600 text-white'
                  : 'bg-red-600 text-white'
              }`}
            >
              {isVerified ? (
                <CheckCircle className="w-8 h-8 flex-shrink-0" />
              ) : (
                <XCircle className="w-8 h-8 flex-shrink-0" />
              )}
              <div>
                <p className="font-bold text-2xl">
                  {result.label}
                </p>
              </div>
            </div>

            {/* Student Info (if available) */}
            {result.student_info && !result.student_info.error && (
              <div className="p-6 bg-white space-y-5">
                <div className="flex items-center gap-4">
                  {result.student_info.passport_url && (
                    <img
                      src={result.student_info.passport_url}
                      alt="Student Passport"
                      className="w-24 h-24 rounded-full object-cover border-4 border-gray-200"
                    />
                  )}
                  <div>
                    <h3 className="text-2xl font-bold text-gray-800">
                      {`${result.student_info.first_name || ''} ${
                        result.student_info.last_name || ''
                      }`}
                    </h3>
                    <p className="text-gray-500 text-lg">
                      {result.student_info.reg_number}
                    </p>
                  </div>
                </div>
                <div className="text-sm space-y-3 pt-4 border-t border-gray-100">
                  <div className="flex justify-between">
                    <span className="text-gray-500">Level:</span>
                    <span className="font-medium text-gray-800">
                      {result.student_info.level}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-500">Full Name:</span>
                    <span className="font-medium text-gray-800">
                      {`${result.student_info.first_name || ''} ${
                        result.student_info.middle_name || ''
                      } ${result.student_info.last_name || ''}`}
                    </span>
                  </div>
                </div>
              </div>
            )}
            
            {/* Analyst Message (Footer) */}
            <div className={`p-5 border-t ${isVerified ? 'bg-green-50 border-green-200 text-green-800' : 'bg-red-50 border-red-200 text-red-800'}`}>
              <p className="font-semibold text-sm">Analyst Note:</p>
              <p className="text-sm">{result.message}</p>
            </div>
            
            {/* Info Error */}
            {result.student_info?.error && (
              <div className="p-4 bg-yellow-50 text-yellow-800 border-t border-yellow-200 text-sm">
                <p className="font-semibold">Information Unavailable:</p>
                <p>{result.student_info.error}</p>
              </div>
            )}
          </div>
        </div>
      );
    }

    // 4. Initial State
    return (
      <div className="flex flex-col items-center justify-center h-full gap-4 text-center p-8 bg-gray-50 rounded-2xl border border-gray-200">
        <h3 className="text-2xl font-bold text-gray-700">
          Verification Panel
        </h3>
        <p className="text-gray-500">
          Upload a document to begin. Your results will appear here.
        </p>
      </div>
    );
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-green-50 to-emerald-100 font-sans flex items-center justify-center p-4 sm:p-6 lg:p-8">
      
      {/* Main App Container */}
      <div className="w-full max-w-6xl bg-white rounded-3xl shadow-2xl overflow-hidden border border-gray-200">
        
        <div className="grid grid-cols-1 md:grid-cols-2">
          
          {/* === LEFT PANEL (ACTION) === */}
          <div className="p-8 lg:p-12 border-r border-gray-100">
            {/* Header */}
            <header className="mb-8 flex items-center gap-4">
              <img
                src="https://www.cesst.org/sites/default/files/styles/large/public/2023-08/unn-logo.png?itok=VLsHF7Z7"
                alt="UNN seal"
                className="w-16 h-16 object-contain"
              />
              <div>
                <h1 className="text-2xl font-extrabold text-gray-900">
                  Handwriting Verification Portal
                </h1>
                <p className="text-sm text-gray-600">
                  University of Nigeria, Nsukka
                </p>
              </div>
            </header>

            {/* Uploader Section */}
            {!previewUrl && (
              <div
                className="border-2 border-dashed border-emerald-300 rounded-2xl p-10 cursor-pointer hover:border-emerald-500 bg-green-50 transition-all duration-300 ease-in-out group"
                onClick={() => document.getElementById('file-input')?.click()}
              >
                <input
                  id="file-input"
                  type="file"
                  accept="image/*"
                  onChange={handleFileChange}
                  className="hidden"
                />
                <div className="flex flex-col items-center text-emerald-700 group-hover:text-emerald-900 transition-colors duration-300">
                  <Upload className="w-12 h-12 mb-4 group-hover:scale-110 transition-transform duration-300" />
                  <p className="font-semibold text-lg">
                    Click to upload document
                  </p>
                  <p className="text-sm text-gray-500 mt-1">
                    PNG, JPG, or WEBP
                  </p>
                </div>
              </div>
            )}

            {/* Preview & Actions Section */}
            {previewUrl && (
              <div className="space-y-6">
                <div className="bg-gray-100 rounded-xl p-3 border border-gray-200 w-full aspect-video flex items-center justify-center overflow-hidden shadow-inner">
                  <img
                    src={previewUrl}
                    alt="Document Preview"
                    className="max-w-full max-h-full object-contain rounded-lg"
                  />
                </div>
                <div className="flex gap-4">
                  <button
                    onClick={handleUpload}
                    disabled={loading}
                    className="flex-1 bg-emerald-700 text-white font-bold py-3 px-6 rounded-xl hover:bg-emerald-800 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-300 ease-in-out flex items-center justify-center gap-2 shadow-md hover:shadow-lg"
                  >
                    {loading ? (
                      <Loader2 className="animate-spin w-5 h-5" />
                    ) : (
                      <FileCheck className="w-5 h-5" />
                    )}
                    <span className="text-lg">Verify Now</span>
                  </button>
                  <button
                    onClick={handleClear}
                    disabled={loading}
                    className="p-3 bg-gray-200 text-gray-700 rounded-xl hover:bg-gray-300 transition-colors duration-300 shadow-sm disabled:opacity-50"
                  >
                    <X className="w-6 h-6" />
                  </button>
                </div>
              </div>
            )}
            
            <footer className="mt-12 text-center text-xs text-gray-500">
              <p>Developed for The Department Of Electronic and Computer Engineering, UNN.</p>
            </footer>
          </div>

          {/* === RIGHT PANEL (RESULT) === */}
          <div className="p-8 lg:p-12 bg-gray-50/50">
            {renderResultPanel()}
          </div>

        </div>
      </div>
    </div>
  );
};

export default App;