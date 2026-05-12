import os
import sys
import pyodbc
import datetime
from typing import Dict, Any
from dotenv import load_dotenv

# Initialize environment variables to securely fetch database credentials
load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

class BookingEngine:
    """
    Handles the appointment registration flow.
    Acts as the direct interface between the AI graph state and the SQL Server 'TblAppointments' table.
    """
    
    @staticmethod
    def register_booking(booking_details: Dict[str, Any]) -> Dict[str, Any]:
        """
        Persists a newly finalized booking request to the live SQL Server.
        Requires a complete set of fields to succeed: property_id, user_name, phone, date.
        """
        # Normalize the date fields since upstream nodes might use alternate terminologies
        if not booking_details.get("date") and booking_details.get("preferred_date"):
            booking_details["date"] = booking_details.get("preferred_date")

        # 1. Validation Gate: Ensure all strict requirements for an appointment are met
        required_booking_fields = ["property_id", "user_name", "phone", "date"]
        missing_booking_fields = [field for field in required_booking_fields if not booking_details.get(field)]
        
        # If the LLM passed an incomplete state, immediately reject the execution
        if missing_booking_fields:
            return {"status": "INCOMPLETE", "missing": missing_booking_fields}
            
        # 2. Connection Assembly: Retrieve connection strings dynamically from the environment
        database_server = os.getenv("DB_SERVER")
        database_name = os.getenv("DB_NAME")
        database_user = os.getenv("DB_USER")
        database_password = os.getenv("DB_PASS")
        database_driver = "{" + os.getenv("DB_DRIVER", "SQL Server") + "}"
        
        sql_connection_string = f"DRIVER={database_driver};SERVER={database_server};DATABASE={database_name};UID={database_user};PWD={database_password};TrustServerCertificate=yes;"
        
        try:
            # Establish direct connection with the target SQL instance
            sql_connection = pyodbc.connect(sql_connection_string)
            sql_cursor = sql_connection.cursor()
            
            # 3. Prepare the Insert Statement
            # Consolidate user identity and contact info into the broad 'Notes' column for the CRM
            # Default the new appointment StatusID to 1 ('Pending/New' state in the downstream application)
            appointment_notes = f"User: {booking_details['user_name']} | Phone: {booking_details['phone']}"
            
            sql_insert_query = """
            INSERT INTO TblAppointments (StatusID, PropertyID, EmployeeClientID, AppointmentDate, Notes)
            VALUES (?, ?, ?, ?, ?)
            """
            
            # Apply strict formatting to the date or default to the current timestamp if mangled
            try:
                # Target format expected by SQL: 'YYYY-MM-DD'
                appointment_date = booking_details['date']
            except KeyError:
                appointment_date = datetime.datetime.now().strftime('%Y-%m-%d')

            # Execute the transactional commit
            sql_cursor.execute(sql_insert_query, (1, booking_details['property_id'], 1, appointment_date, appointment_notes))
            sql_connection.commit()
            
            # Retrieve the auto-incremented Identity token from the SQL Server for receipt generation
            sql_cursor.execute("SELECT @@IDENTITY")
            newly_created_booking_id = sql_cursor.fetchone()[0]
            
            sql_connection.close()
            
            print(f"--- [BOOKING] Successfully registered appointment {newly_created_booking_id} in SQL ---")
            
            # Return the success payload to the AI Graph for narrative processing
            return {"status": "SUCCESS", "booking_id": str(newly_created_booking_id)}

        except Exception as sql_error:
            # Trap fatal SQL or networking errors and report them backward
            print(f"!!! [BOOKING_ERROR] SQL Insertion failed: {sql_error}")
            return {"status": "FAILED", "error": str(sql_error)}

if __name__ == "__main__":
    # Internal Unit Test payload simulating a direct call from the Tool Node
    test_booking_payload = {
        "property_id": "1", # Live property ID verified to exist in TblProperties
        "user_name": "Adham SQL Test",
        "phone": "0123456789",
        "date": "2026-06-15"
    }
    print(BookingEngine.register_booking(test_booking_payload))
