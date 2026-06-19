import os
import sqlite3
from typing import Dict, Any


class BookingEngine:
    """
    Handles the appointment registration flow using the local SQLite database.
    """

    DB_PATH = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "4.Data", "2_DataBase", "RealEstate.db")
    )

    @staticmethod
    def register_booking(booking_details: Dict[str, Any]) -> Dict[str, Any]:
        """
        Persists a newly finalized booking request to SQLite.
        Requires a complete set of fields to succeed: property_id, user_name, phone, date.
        """
        if not booking_details.get("date") and booking_details.get("preferred_date"):
            booking_details["date"] = booking_details.get("preferred_date")

        required_booking_fields = ["property_id", "user_name", "phone", "date"]
        missing_booking_fields = [field for field in required_booking_fields if not booking_details.get(field)]

        if missing_booking_fields:
            return {"status": "INCOMPLETE", "missing": missing_booking_fields}

        try:
            with sqlite3.connect(BookingEngine.DB_PATH) as connection:
                cursor = connection.cursor()
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS TblAppointments (
                        AppointmentID INTEGER PRIMARY KEY AUTOINCREMENT,
                        PropertyID INTEGER NOT NULL,
                        UserName TEXT NOT NULL,
                        Phone TEXT NOT NULL,
                        AppointmentDate TEXT NOT NULL,
                        Notes TEXT,
                        CreatedAt TEXT DEFAULT CURRENT_TIMESTAMP
                    )
                    """
                )
                cursor.execute(
                    """
                    INSERT INTO TblAppointments (PropertyID, UserName, Phone, AppointmentDate, Notes)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        booking_details["property_id"],
                        booking_details["user_name"],
                        booking_details["phone"],
                        booking_details["date"],
                        booking_details.get("notes", ""),
                    ),
                )
                booking_id = cursor.lastrowid

            print(f"--- [BOOKING] Successfully registered appointment {booking_id} in SQLite ---")
            return {"status": "SUCCESS", "booking_id": str(booking_id)}

        except Exception as sqlite_error:
            print(f"!!! [BOOKING_ERROR] SQLite insertion failed: {sqlite_error}")
            return {"status": "FAILED", "error": str(sqlite_error)}


if __name__ == "__main__":
    test_booking_payload = {
        "property_id": "1",
        "user_name": "Adham SQLite Test",
        "phone": "0123456789",
        "date": "2026-06-15",
    }
    print(BookingEngine.register_booking(test_booking_payload))
