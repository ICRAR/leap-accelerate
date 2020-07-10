/**
 * ICRAR - International Centre for Radio Astronomy Research
 * (c) UWA - The University of Western Australia
 * Copyright by UWA(in the framework of the ICRAR)
 * All rights reserved
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place, Suite 330, Boston,
 * MA 02111 - 1307  USA
 */

#pragma once

#include <boost/asio.hpp>
#include <boost/bind.hpp>
#include <boost/function.hpp>

#include <iostream>

using boost::asio::ip::tcp;

namespace icrar
{
    // https://www.boost.org/doc/libs/1_63_0/doc/html/boost_asio/example/cpp03/echo/async_tcp_echo_server.cpp
    class Session
    {
        tcp::socket _socket;
        char _data[1024];
        const int max_length = 1024;

    public:
        Session(boost::asio::io_service& io_service)
        : _socket(io_service)
        {

        }

        tcp::socket& get_socket()
        {
            return _socket;
        }

        void start()
        {
            _socket.async_read_some(boost::asio::buffer(_data, max_length),
                boost::bind(&Session::handle_read, this,
                boost::asio::placeholders::error,
                boost::asio::placeholders::bytes_transferred));
        }

        void handle_read(const boost::system::error_code& error, size_t bytes_transferred)
        {

        }


        void handle_write(const boost::system::error_code& error)
        {

        }
    };

    class Server
    {
        boost::asio::io_service& _io_service;
        tcp::acceptor _acceptor;
    public:
        Server(boost::asio::io_service& io_service, short port)
        : _io_service(io_service)
        , _acceptor(io_service, tcp::endpoint(tcp::v4(), port))
        {
            start_accept();
        }

        /**
         * @brief 
         * 
         */
        void start_accept()
        {
            Session* session = new Session(_io_service);

            _acceptor.async_accept(session->get_socket(), [=](const boost::system::error_code& error)
            {
                handle_accept(session, error);
            });
        }

        void handle_accept(Session* new_session, const boost::system::error_code& error)
        {
            if (!error)
            {
                new_session->start();
            }
            else
            {
                delete new_session;
            }

            start_accept();
        }
    };

    /**
     * @brief Server connection handler
     * 
     */
    void LeapHandleRemoteMS()
    {

    };
}
