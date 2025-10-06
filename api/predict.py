# api/predict.py
from http.server import BaseHTTPRequestHandler
import json
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

# Add current path to Python path for local imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Try to import model functions - will work if model files are available
try:
    from model import predict_single_item, load_model_components
    MODEL_LOADED = True
    print("✅ Model functions imported successfully")
except ImportError as e:
    print(f"⚠️ Model import warning: {e}")
    MODEL_LOADED = False

class Handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        """Handle CORS preflight requests"""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, OPTIONS, GET')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_GET(self):
        """Health check and API information endpoint"""
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()

        health_info = {
            "status": "healthy",
            "service": "MPI Price Predictor API",
            "version": "1.0.0",
            "model_loaded": MODEL_LOADED,
            "timestamp": datetime.now().isoformat(),
            "endpoints": {
                "GET /": "Health check and API information",
                "POST /": "Make price predictions",
                "OPTIONS /": "CORS preflight"
            },
            "usage": {
                "method": "POST",
                "content_type": "application/json",
                "required_fields": [
                    "product_type", 
                    "tg_code", 
                    "country_region", 
                    "country", 
                    "industry"
                ],
                "optional_fields": {
                    "horizon_window": "Number of months to predict (default: 1)"
                }
            }
        }

        response = json.dumps(health_info, indent=2)
        self.wfile.write(response.encode('utf-8'))

    def do_POST(self):
        """Handle price prediction requests"""
        try:
            # Set CORS headers
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()

            # Read and parse request data
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            request_data = json.loads(post_data)

            print(f"📥 Received prediction request")

            # Process the prediction
            result = self.process_prediction_request(request_data)

            # Send response
            response = json.dumps(result, indent=2)
            self.wfile.write(response.encode('utf-8'))

        except json.JSONDecodeError as e:
            self.send_error(400, f"Invalid JSON: {str(e)}")
        except Exception as e:
            error_response = {
                "success": False,
                "error": f"Server error: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
            self.wfile.write(json.dumps(error_response).encode('utf-8'))

    def process_prediction_request(self, data):
        """
        Process prediction request and return results
        
        Args:
            data: Dictionary containing prediction parameters
            
        Returns:
            Dictionary with prediction results or error information
        """
        try:
            # Extract and validate parameters
            product_type = data.get("product_type", "").strip()
            tg_code = data.get("tg_code", "").strip()
            country_region = data.get("country_region", "").strip()
            country = data.get("country", "").strip()
            industry = data.get("industry", "").strip()
            horizon_window = data.get("horizon_window", 1)

            # Validate required parameters
            missing_params = []
            if not product_type:
                missing_params.append("product_type")
            if not tg_code:
                missing_params.append("tg_code")
            if not country_region:
                missing_params.append("country_region")
            if not country:
                missing_params.append("country")
            if not industry:
                missing_params.append("industry")

            if missing_params:
                return {
                    "success": False,
                    "error": f"Missing required parameters: {', '.join(missing_params)}",
                    "timestamp": datetime.now().isoformat()
                }

            # Validate horizon_window
            try:
                horizon_window = int(horizon_window)
                if horizon_window < 1 or horizon_window > 24:
                    return {
                        "success": False,
                        "error": "horizon_window must be between 1 and 24 months",
                        "timestamp": datetime.now().isoformat()
                    }
            except (ValueError, TypeError):
                return {
                    "success": False,
                    "error": "horizon_window must be a valid integer",
                    "timestamp": datetime.now().isoformat()
                }

            # Generate predictions
            predictions = self.generate_predictions(
                product_type, tg_code, country_region, country, 
                industry, horizon_window
            )

            return predictions

        except Exception as e:
            return {
                "success": False,
                "error": f"Prediction processing error: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }

    def generate_predictions(self, product_type, tg_code, country_region, 
                           country, industry, horizon_window):
        """
        Generate price predictions using the model or simulation
        
        Args:
            product_type: Type of product
            tg_code: TG product code
            country_region: Geographic region
            country: Country name
            industry: Industry sector
            horizon_window: Number of months to predict
            
        Returns:
            Dictionary with prediction results
        """
        base_price = 20.0  # Base price for simulation
        predictions = []

        # Generate predictions for each month in the horizon
        for month_offset in range(horizon_window):
            # Calculate year and month
            year = 2024
            month = (month_offset % 12) + 1
            year += month_offset // 12

            # Generate prediction
            if MODEL_LOADED:
                # Use actual model prediction
                try:
                    predicted_price = predict_single_item(
                        product_type=product_type,
                        tg_code=tg_code,
                        country_region=country_region,
                        country=country,
                        industry=industry,
                        year=year,
                        month=month
                    )
                    price_value = float(predicted_price) if predicted_price else None
                except Exception as e:
                    price_value = None
                    print(f"⚠️ Model prediction error: {e}")
            else:
                # Use simulated prediction (linear increase)
                price_value = round(base_price + (month_offset * 0.5), 2)

            # Create prediction object
            prediction = {
                "period": month_offset + 1,
                "year": year,
                "month": month,
                "date": f"{year}-{month:02d}",
                "predicted_price": price_value,
                "currency": "USD"
            }

            # Add error information if prediction failed
            if price_value is None:
                prediction["error"] = "Prediction failed for this period"
            
            predictions.append(prediction)

        # Calculate statistics from successful predictions
        successful_predictions = [p for p in predictions if p.get("predicted_price") is not None]
        
        if successful_predictions:
            prices = [p["predicted_price"] for p in successful_predictions]
            statistics = {
                "min_price": min(prices),
                "max_price": max(prices),
                "avg_price": round(sum(prices) / len(prices), 2),
                "price_trend": "increasing" if len(prices) > 1 and prices[-1] > prices[0] 
                              else "decreasing" if len(prices) > 1 and prices[-1] < prices[0] 
                              else "stable"
            }
        else:
            statistics = {
                "min_price": None,
                "max_price": None,
                "avg_price": None,
                "price_trend": "unknown"
            }

        # Prepare final response
        response = {
            "success": True,
            "model_used": "tensorflow" if MODEL_LOADED else "simulation",
            "horizon_window": horizon_window,
            "total_predictions": len(predictions),
            "successful_predictions": len(successful_predictions),
            "failed_predictions": len(predictions) - len(successful_predictions),
            "input_parameters": {
                "product_type": product_type,
                "tg_code": tg_code,
                "country_region": country_region,
                "country": country,
                "industry": industry
            },
            "price_statistics": statistics,
            "predictions": predictions,
            "timestamp": datetime.now().isoformat()
        }

        # Add note if using simulation
        if not MODEL_LOADED:
            response["note"] = "Using simulated data. Deploy with model files for real predictions."

        return response

def main():
    """Local development server - for testing only"""
    from http.server import HTTPServer
    server = HTTPServer(('localhost', 3000), Handler)
    print("🚀 Development server running at http://localhost:3000")
    print("📡 Endpoints:")
    print("   GET  / - Health check and API info")
    print("   POST / - Make predictions")
    print("   Press Ctrl+C to stop the server")
    server.serve_forever()

if __name__ == "__main__":
    main()
