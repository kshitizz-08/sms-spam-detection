"""User Authentication Module — Clean White & Green Theme.
Simple, user-friendly login interface accessible for non-technical users and examiners.
"""
import hashlib
import hmac
import os
import streamlit as st
import database

def hash_password(password: str, salt: bytes = None) -> str:
    salt = salt or os.urandom(16)
    derived = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 260_000)
    return salt.hex() + "$" + derived.hex()

def verify_password(password: str, stored: str) -> bool:
    try:
        salt_hex, expected_hex = stored.split("$", 1)
        candidate = hash_password(password, bytes.fromhex(salt_hex)).split("$", 1)[1]
        return hmac.compare_digest(candidate, expected_hex)
    except (ValueError, TypeError):
        return False

def login_ui() -> bool:
    if st.session_state.get("username"):
        return True

    # Clean White & Green Styling
    st.markdown("""
        <style>
        /* White & Green Theme */
        .login-wrapper {
            max-width: 480px;
            margin: 40px auto 20px auto;
            background: #FFFFFF;
            border: 1.5px solid #D1FAE5;
            border-radius: 16px;
            padding: 36px 32px;
            box-shadow: 0 10px 25px -5px rgba(5, 150, 105, 0.08), 0 8px 10px -6px rgba(5, 150, 105, 0.04);
            text-align: center;
        }
        .login-badge {
            background-color: #ECFDF5;
            color: #047857;
            border: 1px solid #A7F3D0;
            font-size: 0.8rem;
            font-weight: 700;
            padding: 5px 14px;
            border-radius: 20px;
            display: inline-block;
            margin-bottom: 14px;
            letter-spacing: 0.5px;
        }
        .login-title {
            color: #064E3B;
            font-size: 1.7rem;
            font-weight: 700;
            margin: 0 0 6px 0;
        }
        .login-subtitle {
            color: #4B5563;
            font-size: 0.92rem;
            margin: 0 0 20px 0;
            line-height: 1.5;
        }
        </style>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1.6, 1])
    with col2:
        st.markdown("""
            <div class="login-wrapper">
                <span class="login-badge">🛡️ AI SPAM PROTECTION</span>
                <h1 class="login-title">SMS Spam Shield</h1>
                <p class="login-subtitle">
                    A simple and intelligent system to check if an SMS or text message is safe or a scam.
                </p>
            </div>
        """, unsafe_allow_html=True)

        # One-click quick demo button (ideal for non-technical users and teachers)
        st.markdown("""
            <div style="background: #F0FDF4; border: 1px solid #BBF7D0; border-radius: 10px; padding: 14px; margin-bottom: 16px; text-align: center;">
                <div style="font-size: 0.88rem; color: #166534; font-weight: 600; margin-bottom: 8px;">
                    Just testing the app? No registration needed:
                </div>
            </div>
        """, unsafe_allow_html=True)

        if st.button("👉 Quick Access (Enter App Instantly)", use_container_width=True, type="primary"):
            demo_user = "Guest_User"
            if not database.get_user(demo_user):
                database.add_user(demo_user, hash_password("DemoGuest2026!"))
            st.session_state["username"] = demo_user
            st.rerun()

        st.markdown("<div style='text-align: center; margin: 18px 0; color: #9CA3AF; font-size: 0.85rem;'>— OR SIGN IN WITH USERNAME —</div>", unsafe_allow_html=True)

        tab_login, tab_register = st.tabs(["Sign In", "Create New Account"])

        with tab_login:
            with st.form("login_form"):
                username = st.text_input("Username", placeholder="e.g. alex").strip()
                password = st.text_input("Password", type="password", placeholder="Enter your password")
                submitted = st.form_submit_button("Sign In", use_container_width=True)
                if submitted:
                    if not username or not password:
                        st.error("Please enter both username and password.")
                    else:
                        user = database.get_user(username)
                        if user and verify_password(password, user["password_hash"]):
                            st.session_state["username"] = username
                            st.rerun()
                        else:
                            st.error("Invalid username or password. Or use 'Quick Access' above.")

        with tab_register:
            with st.form("register_form"):
                reg_username = st.text_input("Choose Username", placeholder="e.g. student").strip()
                reg_password = st.text_input("Create Password", type="password", placeholder="At least 8 characters")
                reg_confirm = st.text_input("Confirm Password", type="password", placeholder="Repeat password")
                reg_submitted = st.form_submit_button("Create Account", use_container_width=True)
                if reg_submitted:
                    if not reg_username or not reg_password:
                        st.error("Please fill in all fields.")
                    elif len(reg_password) < 8:
                        st.error("Password must be at least 8 characters.")
                    elif reg_password != reg_confirm:
                        st.error("Passwords do not match.")
                    elif database.get_user(reg_username):
                        st.error("That username already exists. Please choose another.")
                    else:
                        try:
                            database.add_user(reg_username, hash_password(reg_password))
                            st.success("Account created! You can now sign in.")
                        except Exception as e:
                            st.error(f"Error: {e}")

    return False
